import csv
import datetime
import json
import os
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from usage_utils import log_usage  # <-- Use the utility module

USERS_FILE_PATH = "authorized_users.csv"
CONFIG_FILE_PATH = "workers.csv"
LOG_FILE_PATH = "access_log.txt"
MODELS_FILE_PATH = "models.csv"
MODEL_SIZES_FILE_PATH = "model_sizes.csv"
DEACTIVATE_SECURITY = False

# Concurrency-safe global worker state
_STATE_LOCK = threading.Lock()
# name -> {
#   url, queue,
#   vram_total_mb:int|None,
#   loaded_models:set[str], last_models_refresh:float,
#   available_models:set[str], last_available_refresh:float
# }
_WORKERS = {}

def _abs_path(filename):
    return str(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))


def _ensure_model_sizes_file():
    path = _abs_path(MODEL_SIZES_FILE_PATH)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'size_billion'])


def _load_model_sizes():
    _ensure_model_sizes_file()
    df = pd.read_csv(_abs_path(MODEL_SIZES_FILE_PATH))
    sizes = {}
    if not df.empty:
        for _, row in df.iterrows():
            try:
                sizes[str(row['model'])] = float(row['size_billion'])
            except Exception:
                continue
    return sizes


def _save_model_size(model: str, size_billion: float):
    sizes = _load_model_sizes()
    sizes[model] = size_billion
    # atomic-ish write
    tmp_path = _abs_path(MODEL_SIZES_FILE_PATH) + ".tmp"
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'size_billion'])
        for m, s in sizes.items():
            writer.writerow([m, s])
    os.replace(tmp_path, _abs_path(MODEL_SIZES_FILE_PATH))


def _fetch_and_cache_model_size(model: str):
    """Try to get parameter size (e.g., 7B) from /api/show on any healthy worker and cache it."""
    if not model:
        return None
    try:
        _refresh_worker_registry()
        names = _get_enabled_healthy_workers()
        if not names:
            return None
        # Use the first healthy backend to ask about the model
        with _STATE_LOCK:
            n = names[0]
            url = _WORKERS.get(n, {}).get('url')
        if not url:
            return None
        resp = requests.post(f"{url.rstrip('/')}/api/show", json={"model": model}, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json() or {}
        details = data.get('details') or {}
        param_size = details.get('parameter_size') or details.get('parameter_size_str')
        if isinstance(param_size, str):
            m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*([bm])", param_size.strip())
            if m:
                val = float(m.group(1))
                unit = m.group(2).lower()
                size_b = val if unit == 'b' else val / 1000.0
                if size_b > 0:
                    _save_model_size(model, size_b)
                    return size_b
        return None
    except Exception:
        return None


def _parse_model_size_from_string(model: str):
    # Accept forms like: "llama3:8b", "mistral:7B", "tiny:500m", and standalone like "0.5B" or "500m"
    if not isinstance(model, str):
        return None
    s = model.strip()
    # Match optional prefix up to ':', then a number (int or decimal) and unit b/m at the end (case-insensitive)
    m = re.search(r"(?i)(?:.*:)?(\d+(?:\.\d+)?)([bm])$", s)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit == 'b':
        return value
    if unit == 'm':
        return value / 1000.0
    return None


def _estimate_required_vram_mb(size_billion: float | None):
    # Coarse mapping based on common quantized footprints
    # <=7B -> 8GB; <=14B -> 12GB; <=32B -> 24GB; <=70B -> 48GB; else 80GB
    base = 22000
    if size_billion is None:
        return base  # Default conservatively to 24GB
    if size_billion <= 7:
        return 8192
    if size_billion <= 14:
        return base // 2
    if size_billion <= 32:
        return base
    if size_billion <= 70:
        return base * 2
    return 81920


def _vram_tier(total_mb: int | None):
    if total_mb is None:
        return 'unknown'
    if total_mb < 11000:
        return '8-10GB'
    if total_mb < 18000:
        return '12-16GB'
    if total_mb < 30000:
        return '24GB'
    if total_mb < 60000:
        return '48GB'
    return '80GB+'


def _get_workers_df():
    return pd.read_csv(_abs_path(CONFIG_FILE_PATH))


def _save_workers_df(df: pd.DataFrame):
    tmp = _abs_path(CONFIG_FILE_PATH) + ".tmp"
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, _abs_path(CONFIG_FILE_PATH))


def _refresh_worker_registry():
    # Sync _WORKERS with workers.csv; preserve existing queues
    df = _get_workers_df()
    with _STATE_LOCK:
        names_in_csv = set()
        for _, row in df.iterrows():
            name = row.get('name')
            if not isinstance(name, str):
                continue
            url = row.get('url')
            enabled = bool(str(row.get('enabled', True)).strip().lower() == 'true') if isinstance(row.get('enabled', True), str) else bool(row.get('enabled', True))
            healthy = bool(str(row.get('healthy', True)).strip().lower() == 'true') if isinstance(row.get('healthy', True), str) else bool(row.get('healthy', True))
            vram_val = int(row.get('vram_total_mb')) if 'vram_total_mb' in df.columns and pd.notna(row.get('vram_total_mb')) else None

            if name not in _WORKERS:
                _WORKERS[name] = {
                    'url': url,
                    'queue': Queue(),
                    'vram_total_mb': vram_val,
                    'loaded_models': set(),
                    'last_models_refresh': 0.0,
                    'available_models': set(),
                    'last_available_refresh': 0.0,
                    'enabled': enabled,
                    'healthy': healthy,
                }
            else:
                _WORKERS[name]['url'] = url
                # carry queue
                if 'queue' not in _WORKERS[name] or not isinstance(_WORKERS[name]['queue'], Queue):
                    _WORKERS[name]['queue'] = Queue()
                # Update VRAM cached
                if vram_val is not None:
                    _WORKERS[name]['vram_total_mb'] = vram_val
                _WORKERS[name]['enabled'] = enabled
                _WORKERS[name]['healthy'] = healthy

            names_in_csv.add(name)

        # Remove any workers no longer in CSV
        for name in list(_WORKERS.keys()):
            if name not in names_in_csv:
                del _WORKERS[name]


def _fetch_and_cache_worker_vram_if_missing(name: str):
    # Try to get vram_total_mb from CSV or helper API; persist to CSV
    with _STATE_LOCK:
        w = _WORKERS.get(name)
    if not w:
        return
    if w.get('vram_total_mb'):
        return
    # Load CSV row to check enabled+healthy and URL
    df = _get_workers_df()
    row = df[df['name'] == name]
    if row.empty:
        return
    row = row.iloc[0]
    enabled = bool(str(row.get('enabled', True)).strip().lower() == 'true') if isinstance(row.get('enabled', True), str) else bool(row.get('enabled', True))
    healthy = bool(str(row.get('healthy', True)).strip().lower() == 'true') if isinstance(row.get('healthy', True), str) else bool(row.get('healthy', True))
    if not (enabled and healthy):
        return
    url = str(row.get('url'))
    try:
        helper_url = f"{url.replace('11434','18034').rstrip('/')}/gpu/vram"
        resp = requests.get(helper_url, headers={"x-api-key": os.environ.get("OLLAMA_HELPER_API_KEY", "")}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            total = int(data.get('vram_total_mb', 0))
            if total > 0:
                with _STATE_LOCK:
                    _WORKERS[name]['vram_total_mb'] = total
                # Persist to CSV
                df.loc[df['name'] == name, 'vram_total_mb'] = total
                _save_workers_df(df)
    except Exception:
        pass


def _refresh_loaded_models(name: str, ttl_seconds: int = 30):
    now = time.time()
    with _STATE_LOCK:
        w = _WORKERS.get(name)
        if not w:
            return set()
        if now - w.get('last_models_refresh', 0) < ttl_seconds and w.get('loaded_models') is not None:
            return set(w['loaded_models'])
        url = w.get('url')
    try:
        resp = requests.get(f"{url.rstrip('/')}/api/ps", timeout=4)
        models = set()
        if resp.status_code == 200:
            data = resp.json()
            for m in data.get('models', []) or []:
                n = m.get('name')
                if isinstance(n, str) and n:
                    models.add(n)
        with _STATE_LOCK:
            if name in _WORKERS:
                _WORKERS[name]['loaded_models'] = models
                _WORKERS[name]['last_models_refresh'] = now
        return models
    except Exception:
        with _STATE_LOCK:
            return set(_WORKERS.get(name, {}).get('loaded_models') or set())


def _refresh_available_models(name: str, ttl_seconds: int = 60):
    now = time.time()
    with _STATE_LOCK:
        w = _WORKERS.get(name)
        if not w:
            return set()
        if now - w.get('last_available_refresh', 0) < ttl_seconds and w.get('available_models') is not None:
            return set(w['available_models'])
        url = w.get('url')
    try:
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        available = set()
        if resp.status_code == 200:
            data = resp.json()
            # Newer ollama: {"models":[{"name":"..."}, ...]}
            models_list = []
            if isinstance(data, dict) and 'models' in data:
                models_list = data.get('models') or []
            elif isinstance(data, list):
                models_list = data
            for m in models_list:
                try:
                    n = m.get('name') if isinstance(m, dict) else None
                    if isinstance(n, str) and n:
                        available.add(n)
                except Exception:
                    continue
        with _STATE_LOCK:
            if name in _WORKERS:
                _WORKERS[name]['available_models'] = available
                _WORKERS[name]['last_available_refresh'] = now
        return available
    except Exception:
        with _STATE_LOCK:
            return set(_WORKERS.get(name, {}).get('available_models') or set())


def _get_enabled_healthy_workers():
    df = _get_workers_df()
    candidates = []
    for _, row in df.iterrows():
        try:
            enabled = bool(str(row.get('enabled', True)).strip().lower() == 'true') if isinstance(row.get('enabled', True), str) else bool(row.get('enabled', True))
            healthy = bool(str(row.get('healthy', True)).strip().lower() == 'true') if isinstance(row.get('healthy', True), str) else bool(row.get('healthy', True))
            if not (enabled and healthy):
                continue
            name = row['name']
            candidates.append(name)
        except Exception:
            continue
    return candidates


def init_worker_state_cache():
    """Synchronously warm the worker cache for VRAM, loaded and available models.
    Safe to call at startup before handling requests.
    """
    try:
        _refresh_worker_registry()
        names = _get_enabled_healthy_workers()
        for n in names:
            try:
                _fetch_and_cache_worker_vram_if_missing(n)
            except Exception:
                pass
            try:
                # Force-refresh both caches with TTL=0 to ensure immediate population
                _refresh_available_models(n, ttl_seconds=0)
            except Exception:
                pass
            try:
                _refresh_loaded_models(n, ttl_seconds=0)
            except Exception:
                pass
    except Exception as e:
        print(f"init_worker_state_cache error: {e}")


def _choose_backend_for_model(model: str):
    # Refresh registry
    _refresh_worker_registry()

    # Ensure VRAM cached for candidates (best effort, non-blocking for requests thread)
    names = _get_enabled_healthy_workers()
    for n in names:
        try:
            _fetch_and_cache_worker_vram_if_missing(n)
        except Exception:
            pass

    # Model size if provided
    sizes_cache = _load_model_sizes()
    size_b = sizes_cache.get(model)
    if size_b is None:
        parsed = _parse_model_size_from_string(model)
        if parsed is not None:
            size_b = parsed
            _save_model_size(model, size_b)
        else:
            size_b = _fetch_and_cache_model_size(model)
    required_vram = _estimate_required_vram_mb(size_b)

    # Build list of worker info tuples using ONLY cached state
    # (name, url, q, vram_total_mb, loaded_models, vram_tier, available_models)
    def _snapshot():
        info = []
        with _STATE_LOCK:
            for n in names:
                w = _WORKERS.get(n)
                if not w:
                    continue
                url = w.get('url')
                q = w.get('queue')
                vram = w.get('vram_total_mb')
                loaded_cached = set(w.get('loaded_models') or [])
                available_cached = set(w.get('available_models') or [])
                info.append((n, url, q, vram, loaded_cached, _vram_tier(vram), available_cached))
        return info

    workers_info = _snapshot()

    # If a model is specified, restrict candidates to those that have it available
    if model:
        filtered = [w for w in workers_info if model in w[6] or model in w[4]]
        if not filtered:
            # Force a one-time refresh of available models to avoid cold-start misses
            for n in names:
                try:
                    _refresh_available_models(n, ttl_seconds=0)
                except Exception:
                    pass
            workers_info = _snapshot()
            filtered = [w for w in workers_info if model in w[6] or model in w[4]]
        workers_info = filtered
        if not workers_info:
            return None

    # A) Model already loaded somewhere
    loaded_workers = [(n, url, q, vram, tier) for (n, url, q, vram, loaded, tier, avail) in workers_info if model and model in loaded]
    if loaded_workers:
        empties = [w for w in loaded_workers if w[2].qsize() == 0]
        if empties:
            chosen = sorted(empties, key=lambda x: ((x[3] or 1_000_000), x[2].qsize()))[0]
            return chosen
        # 2) try same-tier empty, without any loaded models yet, but with the model available
        tiers = {w[4] for w in loaded_workers}
        same_tier_candidates = [w for w in workers_info if w[5] in tiers and len(w[4]) == 0 and w[2].qsize() == 0 and (not model or model in w[6])]
        if same_tier_candidates:
            chosen = sorted(same_tier_candidates, key=lambda x: ((x[3] or 1_000_000), x[2].qsize()))[0]
            return (chosen[0], chosen[1], chosen[2], chosen[3], chosen[5])
        chosen = sorted(loaded_workers, key=lambda x: x[2].qsize())[0]
        return chosen

    # B) Model not loaded: choose by fit, availability, and shortest queue
    fitting = [w for w in workers_info if (w[3] or 0) >= required_vram]
    if model:
        fitting = [w for w in fitting if model in w[6]]

    if not fitting:
        # Fallback: prefer any with model availability first even if VRAM insufficient, else any
        if workers_info:
            candidates = [w for w in workers_info if (not model or model in w[6])]
            if not candidates:
                candidates = workers_info
            best = sorted(candidates, key=lambda x: (-(x[3] or 0), x[2].qsize()))[0]
            return (best[0], best[1], best[2], best[3], best[5])
        return None

    min_vram = min(w[3] for w in fitting if w[3] is not None)
    close_fit = [w for w in fitting if w[3] == min_vram]
    best = sorted(close_fit, key=lambda x: x[2].qsize())[0]
    return (best[0], best[1], best[2], best[3], best[5])


#ollama api endpoints:
# POST /api/generate
# POST /api/chat
# POST /api/embed
# POST /api/embeddings (deprecated in favor of /api/embed)
# POST /api/pull
# POST /api/push
# POST /api/create
# POST /api/blobs/{digest}
# GET /api/tags
# DELETE /api/delete
# POST /api/copy
# POST /api/show
# GET /api/ps

# Still needs better server choice if multiple servers have same queue length or model does not fit on one specific server.


def get_user_key(user):
    users = pd.read_csv(_abs_path(USERS_FILE_PATH))
    users = dict(zip(users['user'], users['accessKey']))
    return users.get(user, None)


# The previous get_config is replaced by a registry-backed approach


def _save_last_used(model):
    today = datetime.datetime.today().strftime("%d.%m.%Y")
    rows = []
    models = _abs_path(MODELS_FILE_PATH)
    with open(models, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) == 2 and row[0] == model:
                row[1] = today
            rows.append(row)

    with open(models, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class RequestHandler(BaseHTTPRequestHandler):

    def add_access_log_entry(self, event, user, ip_address, access, server, nb_queued_requests_on_server, error=""):
        # Uses global LOG_FILE_PATH
        log_file_path_obj = Path(LOG_FILE_PATH)

        if not log_file_path_obj.exists():
            with open(log_file_path_obj, mode='w', newline='') as csvfile:
                fieldnames = ['time_stamp', 'event', 'user_name', 'ip_address', 'access', 'server', 'nb_queued_requests_on_server', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(log_file_path_obj, mode='a', newline='') as csvfile:
            fieldnames = ['time_stamp', 'event', 'user_name', 'ip_address', 'access', 'server', 'nb_queued_requests_on_server', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row = {'time_stamp': str(datetime.datetime.now()), 'event':event, 'user_name': user, 'ip_address': ip_address, 'access': access, 'server': server, 'nb_queued_requests_on_server': nb_queued_requests_on_server, 'error': error}
            writer.writerow(row)


    def _send_response(self, response):
        self.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() not in ['content-length', 'transfer-encoding', 'content-encoding']:
                self.send_header(key, value)
        self.end_headers()

        try:
            content = response.content
            if hasattr(response, 'iter_content'):
                for chunk in response.iter_content(chunk_size=8192):
                    self.wfile.write(chunk)
            else:
                self.wfile.write(content)
            self.wfile.flush()
        except BrokenPipeError:
            print(f"Broken pipe error for {self.client_address}")
            pass
        except Exception as e:
            print(f"Error sending response content: {e}")


    def do_HEAD(self):
        self.log_request()
        self.proxy()


    def do_GET(self):
        self.log_request()
        self.proxy()


    def do_POST(self):
        self.log_request()
        self.proxy()


    def _validate_user_and_key(self):
        try:
            auth_header = self.headers.get('Authorization')
            if not auth_header:
                return False
            token = auth_header
            user, key = token.split(':', 1)

            if get_user_key(user) == key:
                self.user = user
                return True
            else:
                self.user = "unknown (failed_auth)"
            return False
        except Exception as e:
            print(f"Auth validation error: {e}")
            self.user = "unknown (auth_error)"
            return False


    def proxy(self):
        self.user = "unknown"
        client_ip, client_port = self.client_address
        if not DEACTIVATE_SECURITY and not self._validate_user_and_key():
            print(f'User is not authorized from {client_ip}:{client_port}')
            auth_header = self.headers.get('Authorization')
            token_info = "No token"
            if auth_header:
                token_info = auth_header
            self.add_access_log_entry(event='rejected', user=token_info, ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="Authentication failed")
            self.send_response(403)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Forbidden: Authentication failed"}).encode('utf-8'))
            return
        print(f"User '{self.user}' from {client_ip}:{client_port} is authorized.")
        url = urlparse(self.path)
        path = url.path
        get_params = parse_qs(url.query) or {}

        post_data = b''
        if self.command == "POST":
            print(f"POST request to {path} from user {self.user}")
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
            except (TypeError, ValueError):
                print("POST request without valid Content-Length.")
                pass

        # Allow all GET requests to be proxied to first healthy backend (for non-model endpoints)
        if self.command == "GET":
            print(f"GET request to {path} from user {self.user}")
            _refresh_worker_registry()
            names = _get_enabled_healthy_workers()
            if not names:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Service Unavailable: No backend servers configured."}).encode('utf-8'))
                return
            with _STATE_LOCK:
                first_name = names[0]
                target_url = _WORKERS[first_name]['url']
            try:
                response = requests.request(
                    self.command,
                    target_url + path,
                    params=get_params,
                    headers={k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']}
                )
                self._send_response(response)
            except requests.exceptions.RequestException as ex:
                print(f"Proxy GET request to {first_name} failed: {ex}")
                self.send_response(502)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
            except Exception as ex_other:
                print(f"Unexpected error during proxy GET to {first_name}: {ex_other}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))
            return

        # Only allow certain POST endpoints
        allowed_post_paths = [
            '/api/generate',
            '/api/chat',
            '/api/embed',
            '/api/embeddings',
            '/api/show',
            '/v1/chat/completions'
        ]

        if self.command == "POST" and path in allowed_post_paths:
            # Figure out model if present
            print(f"POST request to {path} from user {self.user}")
            model = None
            is_streaming = False
            if post_data:
                try:
                    post_data_dict = json.loads(post_data.decode('utf-8'))
                    model = post_data_dict.get('model')
                    is_streaming = post_data_dict.get('stream', False)
                    if model:
                        _save_last_used(model)
                except Exception:
                    pass
            print(f"Requested model: {model if model else 'None'}")
            chosen = _choose_backend_for_model(model or "")
            print(f"Chosen backend for model '{model}': {chosen[0] if chosen else 'None'}")
            if not chosen:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Service Unavailable: No suitable backend available."}).encode('utf-8'))
                return

            name, target_url, que, _, _ = chosen
            self.add_access_log_entry(event="gen_request", user=self.user, ip_address=client_ip, access="Authorized", server=name, nb_queued_requests_on_server=que.qsize())
            que.put_nowait(1)
            start_time = time.time()
            try:
                print(f"Proxying {self.command} {path} to {name} at {target_url} for user {self.user}")
                response = requests.request(
                    self.command,
                    target_url + path,
                    params=get_params,
                    data=post_data,
                    headers={k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']},
                    stream=is_streaming
                )
                self._send_response(response)
            except requests.exceptions.RequestException as ex:
                print(f"Proxy request to {name} failed: {ex}")
                traceback.print_exc()
                self.add_access_log_entry(event="gen_error",user=self.user, ip_address=client_ip, access="Authorized", server=name, nb_queued_requests_on_server=que.qsize(),error=str(ex))
                self.send_response(502)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
            except Exception as ex_other:
                print(f"Unexpected error during proxy to {name}: {ex_other}")
                traceback.print_exc()
                self.add_access_log_entry(event="gen_error",user=self.user, ip_address=client_ip, access="Authorized", server=name, nb_queued_requests_on_server=que.qsize(),error=str(ex_other))
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))
            finally:
                duration = time.time() - start_time
                if self.user and self.user != "unknown" and duration > 0:
                    try:
                        log_usage(self.user, duration)
                    except Exception as e:
                        print(f"Failed to log usage for user {self.user}: {e}")
                if not que.empty():
                    que.get_nowait()
                self.add_access_log_entry(event="gen_done",user=self.user, ip_address=client_ip, access="Authorized", server=name, nb_queued_requests_on_server=que.qsize())
            return

        # Block all other POST endpoints with an error message
        if self.command == "POST":
            _refresh_worker_registry()
            names = _get_enabled_healthy_workers()
            server_name = names[0] if names else "None"
            self.add_access_log_entry(
                event='blocked_post',
                user=self.user,
                ip_address=client_ip,
                access="Denied",
                server=server_name,
                nb_queued_requests_on_server=0,
                error=f"Blocked POST to {path}"
            )
            self.send_response(403)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Forbidden: POST to '{path}' is not allowed. Allowed POST endpoints: {', '.join(allowed_post_paths)}"
            }).encode('utf-8'))
            return

        # For any other method, fallback to simple proxying to first backend
        _refresh_worker_registry()
        names = _get_enabled_healthy_workers()
        if not names:
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Service Unavailable: No backend servers configured."}).encode('utf-8'))
            return
        with _STATE_LOCK:
            first_name = names[0]
            target_url = _WORKERS[first_name]['url']
        try:
            response = requests.request(
                self.command,
                target_url + path,
                params=get_params,
                data=post_data,
                headers={k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']}
            )
            self._send_response(response)
        except requests.exceptions.RequestException as ex:
            print(f"Proxy request to {first_name} for non-gen endpoint failed: {ex}")
            self.send_response(502)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
        except Exception as ex_other:
            print(f"Unexpected error during proxy (non-gen) to {first_name}: {ex_other}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def _worker_state_refresher_loop(interval_seconds: int = 15, models_ttl_seconds: int = 30):
    """Background loop to keep _WORKERS cache warm (models and VRAM).
    Periodically:
    - refresh the worker registry
    - ensure VRAM totals are cached
    - refresh loaded models with a TTL
    - refresh available models with a TTL
    """
    while True:
        try:
            _refresh_worker_registry()
            names = _get_enabled_healthy_workers()
            for n in names:
                try:
                    _fetch_and_cache_worker_vram_if_missing(n)
                except Exception:
                    pass
                try:
                    _refresh_loaded_models(n, ttl_seconds=models_ttl_seconds)
                except Exception:
                    pass
                try:
                    _refresh_available_models(n, ttl_seconds=max(models_ttl_seconds, 45))
                except Exception:
                    pass
        except Exception as e:
            print(f"Worker state refresher error: {e}")
        time.sleep(max(5, int(interval_seconds)))


def start_worker_state_refresher_thread(interval_seconds: int = 15, models_ttl_seconds: int = 30):
    t = threading.Thread(target=_worker_state_refresher_loop, kwargs={
        'interval_seconds': interval_seconds,
        'models_ttl_seconds': models_ttl_seconds,
    }, daemon=True)
    t.start()
    return t
