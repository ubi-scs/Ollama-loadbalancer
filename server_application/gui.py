import argparse
import csv
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from tempfile import NamedTemporaryFile

import gradio as gr
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from usage_utils import log_usage

from proxy import (
    RequestHandler,
    ThreadedHTTPServer,
    LOG_FILE_PATH,
    start_worker_state_refresher_thread,
    init_worker_state_cache,
)


CORRECT_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not CORRECT_PASSWORD:
    raise RuntimeError(
        "ADMIN_PASSWORD environment variable not set. Please provide a secret Admin password."
    )

OLLAMA_HELPER_API_KEY = os.environ.get("OLLAMA_HELPER_API_KEY")
if not OLLAMA_HELPER_API_KEY:
    raise RuntimeError(
        "OLLAMA_HELPER_API_KEY environment variable not set. Please provide a secret OLLAMA_HELPER_API_KEY."
    )


WORKER_CONFIG_PATH = "workers.csv"
AUTHORIZED_USERS_CONFIG_PATH = "authorized_users.csv"
MODELS_CONFIG_PATH = "models.csv"
USAGE_STATS_PATH = "usage_stats.json"
MODEL_SIZES_PATH = "model_sizes.csv"


def _abs_path(local_path: str) -> str:
    return str(os.path.join(os.path.dirname(os.path.abspath(__file__)), local_path))


def _parse_model_size_from_string(model: str):
    """Parse size like ':7b', ':0.5B', ':500m' from model string. Returns billions (float) or None."""
    if not isinstance(model, str):
        return None
    s = model.strip()
    if s.startswith("-"):
        return None
    colon_idx = s.rfind(":")
    if colon_idx >= 0 and s[colon_idx + 1 :].lstrip().startswith("-"):
        return None
    m = re.search(r"(?i)(?:.*:)?(\d+(?:\.\d+)?)([bm])$", s)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if value <= 0:
        return None
    unit = m.group(2).lower()
    if unit == "b":
        return value
    if unit == "m":
        return value / 1000.0
    return None


def _estimate_required_vram_mb(size_billion: float | None):
    base = 22000
    if size_billion is None or size_billion <= 0:
        return base
    if size_billion <= 7:
        return 8192
    if size_billion <= 14:
        return base // 2
    if size_billion <= 32:
        return base
    if size_billion <= 70:
        return base * 2
    return 81920


def _ensure_model_sizes_file():
    path = _abs_path(MODEL_SIZES_PATH)
    if not os.path.exists(path):
        pd.DataFrame(columns=["model", "size_billion"]).to_csv(
            path, index=False, encoding="utf-8"
        )


def _save_model_size(model: str, size_billion: float):
    try:
        _ensure_model_sizes_file()
        path = _abs_path(MODEL_SIZES_PATH)
        df = pd.read_csv(path)
        if "model" not in df.columns or "size_billion" not in df.columns:
            df = pd.DataFrame(columns=["model", "size_billion"])
        if not df.empty and (df["model"] == model).any():
            df.loc[df["model"] == model, "size_billion"] = size_billion
        else:
            df = pd.concat(
                [df, pd.DataFrame([{"model": model, "size_billion": size_billion}])],
                ignore_index=True,
            )
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        print(f"GUI: Failed to save model size for {model}: {e}")


def _get_worker_vram_total_mb(url: str, name: str, df: pd.DataFrame) -> int | None:
    total = None
    try:
        if "vram_total_mb" in df.columns:
            row = df[df["name"] == name]
            if not row.empty:
                val = row.iloc[0].get("vram_total_mb")
                if pd.notna(val):
                    try:
                        total = int(float(val))
                    except Exception:
                        total = None
        if not total or total <= 0:
            helper_url = _rewrite_url_to_helper_port(str(url))
            if helper_url is None:
                return total
            helper_url = f"{helper_url.rstrip('/')}/gpu/vram"
            resp = requests.get(
                helper_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json() or {}
                t = int(data.get("vram_total_mb", 0))
                if t > 0:
                    total = t
                    try:
                        df.loc[df["name"] == name, "vram_total_mb"] = int(t)
                        path = (
                            WORKER_CONFIG_PATH
                            if os.path.isabs(WORKER_CONFIG_PATH)
                            else _abs_path(WORKER_CONFIG_PATH)
                        )
                        tmp = path + ".tmp"
                        df.to_csv(tmp, index=False, encoding="utf-8")
                        os.replace(tmp, path)
                    except Exception as e:
                        print(f"GUI: Failed to persist vram_total_mb for {name}: {e}")
    except Exception as e:
        print(f"GUI: Error fetching VRAM total for {name}: {e}")
    return total


def get_worker_status():
    status_data = []

    workers = pd.read_csv(WORKER_CONFIG_PATH)

    # Ensure healthy column exists to prevent KeyError and allow skipping offline workers
    if "healthy" not in workers.columns:
        workers["healthy"] = True

    try:
        for _, row in workers.iterrows():
            worker_url = row["url"]
            worker_name = row["name"]
            enabled = _coerce_bool(row.get("enabled", True), True)
            healthy = _coerce_bool(row.get("healthy", True), True)

            if not enabled:
                status_data.append(
                    [
                        worker_name,
                        worker_url,
                        "disabled",
                        "disabled",
                        "disabled",
                        "disabled",
                        "disabled",
                        enabled,
                    ]
                )
                continue

            # Quick probe to avoid UI timeouts if the healthy flag is stale
            if healthy:
                try:
                    if not check_worker_availability(worker_url):
                        healthy = False
                except Exception:
                    healthy = False

            # If worker is marked unhealthy, avoid network calls and show as offline
            if not healthy:
                status_data.append(
                    [
                        worker_name,
                        worker_url,
                        "offline",
                        "offline",
                        "offline",
                        "offline",
                        "offline",
                        enabled,
                    ]
                )
                continue

            running_models_str = "N/A"  # Default
            api_ps_url = f"{worker_url.rstrip('/')}/api/ps"

            try:
                response = requests.get(api_ps_url, timeout=5)
                if response.status_code == 200:
                    ps_data = response.json()
                    if (
                        ps_data
                        and "models" in ps_data
                        and isinstance(ps_data["models"], list)
                    ):
                        model_names = [
                            model.get("name", "UnknownModel")
                            for model in ps_data["models"]
                        ]
                        if model_names:
                            running_models_str = "; ".join(model_names)
                        else:
                            running_models_str = "None"
                    else:
                        running_models_str = "No model data"
                elif response.status_code == 404:
                    running_models_str = "Unsupported (old Ollama?)"
                else:
                    running_models_str = f"Error: HTTP {response.status_code}"
            except Exception as e_ps:
                running_models_str = "Fetch Error"
                print(f"GUI: Error fetching /api/ps for server {worker_name}: {e_ps}")

            try:
                gpu_util_url = _rewrite_url_to_helper_port(worker_url)
                if gpu_util_url is None:
                    gpu_utilization_str = "N/A (invalid URL)"
                else:
                    gpu_util_url = f"{gpu_util_url}/gpu/utilization"
                    response = requests.get(
                        gpu_util_url,
                        headers={"x-api-key": OLLAMA_HELPER_API_KEY},
                        timeout=5,
                    )
                    if response.status_code == 200:
                        util_data = response.json()
                        gpu_utilization_str = (
                            f"{util_data.get('gpu_utilization_percent', 0.0):.1f}%"
                        )
                    else:
                        gpu_utilization_str = f"Err {response.status_code}"
            except Exception as e_util:
                gpu_utilization_str = "Fetch Error"
                print(
                    f"GUI: Error fetching GPU utilization for server {worker_name}: {e_util}"
                )

                # Fetch VRAM usage
            try:
                vram_url = _rewrite_url_to_helper_port(worker_url)
                if vram_url is None:
                    vram_usage_str = "N/A (invalid URL)"
                else:
                    vram_url = f"{vram_url}/gpu/vram"
                    response = requests.get(
                        vram_url,
                        headers={"x-api-key": OLLAMA_HELPER_API_KEY},
                        timeout=5,
                    )
                    if response.status_code == 200:
                        vram_data = response.json()
                        used = vram_data.get("vram_used_mb", 0)
                        total = vram_data.get("vram_total_mb", 1)
                        vram_usage_str = f"{int(used)}/{int(total)}MB"
                    else:
                        vram_usage_str = f"Err {response.status_code}"
            except Exception as e_vram:
                vram_usage_str = "Fetch Error"
                print(
                    f"GUI: Error fetching VRAM info for server {worker_name}: {e_vram}"
                )

                # Fetch version
            try:
                api_version_url = f"{worker_url.rstrip('/')}/api/version"
                response = requests.get(api_version_url, timeout=5)
                if response.status_code == 200:
                    version_data = response.json()
                    ollama_version = version_data.get("version", "JSON Error")
                else:
                    ollama_version = f"Err {response.status_code}"
            except Exception as e_version:
                ollama_version = "Fetch Error"
                print(
                    f"GUI: Error fetching Ollama version info for server {row['name']}: {e_version}"
                )

            # Fetch activity status from worker helper API
            activity_str = "Available"
            try:
                activity_url = _rewrite_url_to_helper_port(worker_url)
                if activity_url is None:
                    activity_str = "N/A (invalid URL)"
                else:
                    activity_url = f"{activity_url}/worker/activity-status"
                    activity_resp = requests.get(
                        activity_url,
                        headers={"x-api-key": OLLAMA_HELPER_API_KEY},
                        timeout=5,
                    )
                    if activity_resp.status_code == 200:
                        activity_data = activity_resp.json()
                        is_disabled = activity_data.get("disabled", False)
                        if is_disabled:
                            reason = activity_data.get("last_disable_reason", "")
                            remaining = activity_data.get("remaining_seconds", 0)
                            if reason == "user_active":
                                activity_str = f"Self-disabled (user active, {int(remaining)}s remaining)"
                            elif reason == "gpu_vram_contended":
                                activity_str = f"Self-disabled (GPU busy, {int(remaining)}s remaining)"
                            else:
                                activity_str = f"Self-disabled ({reason}, {int(remaining)}s remaining)"
                        else:
                            user_active = activity_data.get("user_active", False)
                            gpu_contended = activity_data.get(
                                "gpu_vram_contended", False
                            )
                            if user_active:
                                activity_str = "User active"
                            elif gpu_contended:
                                activity_str = "GPU contended"
                            else:
                                activity_str = "Available"
                    else:
                        activity_str = f"Err {activity_resp.status_code}"
            except Exception as e_activity:
                activity_str = "N/A"
                print(
                    f"GUI: Error fetching activity status for server {worker_name}: {e_activity}"
                )

            status_data.append(
                [
                    worker_name,
                    worker_url,
                    running_models_str,
                    gpu_utilization_str,
                    vram_usage_str,
                    ollama_version,
                    activity_str,
                    enabled,
                ]
            )

        if not status_data:
            print(
                "GUI: current_servers_config was present, but status_data is empty. Returning default empty row."
            )
            return [["No valid server data found.", "", "", "", ""]]
        return status_data
    except Exception as e:
        print(f"GUI Error: Exception in get_server_status_for_gui: {e}")
        import traceback

        traceback.print_exc()
        return [["Error processing server data.", str(e), "", "", ""]]


# NEW: Health monitoring utilities
def ensure_workers_csv_has_healthy_column():
    path = _abs_path(WORKER_CONFIG_PATH)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return
    if "healthy" not in df.columns:
        df["healthy"] = True
        df.to_csv(path, index=False, encoding="utf-8")


def _coerce_bool(val, default=True):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    return default


def check_worker_availability(url):
    try:
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            print(f"Health check skipped: invalid url '{url}'")
            return False
        # Use the URL exactly as configured; include API key header
        target = f"{url.rstrip('/')}/api/version"
        resp = requests.get(
            target, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=3
        )
        if resp.status_code == 200:
            # basic sanity check on json
            try:
                _ = resp.json()
            except Exception:
                pass
            return True
        else:
            print(f"Health check failed for {target}: HTTP {resp.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"Health check timeout for {url}")
        return False
    except Exception as e:
        print(f"Health check exception for {url}: {e}")
        return False


def health_monitor_loop(interval_seconds=60):
    while True:
        try:
            df = pd.read_csv(WORKER_CONFIG_PATH)
            changed = False
            # ensure column exists
            if "healthy" not in df.columns:
                df["healthy"] = True
                changed = True
            for idx, row in df.iterrows():
                enabled = _coerce_bool(row.get("enabled", True), True)
                url = row.get("url")
                prev_healthy = _coerce_bool(row.get("healthy", True), True)
                # Only probe enabled workers; disabled ones are marked healthy by default to avoid hiding them
                new_healthy = (
                    check_worker_availability(url) if enabled else prev_healthy
                )
                if bool(prev_healthy) != bool(new_healthy):
                    print(
                        f"Health state change for {row.get('name')}: {prev_healthy} -> {new_healthy}"
                    )
                    df.at[idx, "healthy"] = bool(new_healthy)
                    changed = True
            if changed:
                # atomic-ish write
                tmp = NamedTemporaryFile(mode="w", delete=False)
                try:
                    df.to_csv(tmp.name, index=False, encoding="utf-8")
                    os.replace(tmp.name, WORKER_CONFIG_PATH)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Health monitor error: {e}")
        time.sleep(interval_seconds)


def start_health_monitor_thread(interval_seconds=60):
    ensure_workers_csv_has_healthy_column()
    t = threading.Thread(
        target=health_monitor_loop,
        kwargs={"interval_seconds": interval_seconds},
        daemon=True,
    )
    t.start()
    return t


def get_logs(num_lines):
    """
    Reads the last N lines from the access log file.
    """
    log_file_path = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_PATH)
    )

    try:
        if not os.path.exists(log_file_path):
            return "Log file not found."
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_index = max(0, len(lines) - num_lines)
        log_content = "".join(lines[start_index:])

        if not log_content.strip():
            return "Log file is empty or contains only whitespace."

        return log_content
    except Exception as e:
        print(f"GUI Error: Exception in get_access_logs_for_gui: {e}")
        import traceback

        traceback.print_exc()
        return f"Error reading log file: {str(e)}"


def get_global_models():
    if not os.path.exists(MODELS_CONFIG_PATH):
        print("GUI Warning: Models config file not found. Creating new file.")
        pd.DataFrame(columns=["Model", "LastUsed"]).to_csv(
            MODELS_CONFIG_PATH, index=False, encoding="utf-8"
        )
    models = pd.read_csv(MODELS_CONFIG_PATH)
    if not models.empty and "LastUsed" in models.columns:
        models["_sort_date"] = pd.to_datetime(
            models["LastUsed"], format="%d.%m.%Y", errors="coerce"
        )
        models = models.sort_values(
            by="_sort_date", ascending=False, na_position="last"
        )
        models = models.drop(columns=["_sort_date"]).reset_index(drop=True)
    return models


def get_worker_models():
    worker_status_list = []

    # Step 1: read worker info directly from config to respect healthy flag
    try:
        workers_df = pd.read_csv(WORKER_CONFIG_PATH)
    except Exception:
        workers_df = pd.DataFrame(columns=["name", "url", "enabled", "healthy"])

    if "healthy" not in workers_df.columns:
        workers_df["healthy"] = True

    # Step 2: load global models
    global_models_df = get_global_models()
    if global_models_df is None or "Model" not in global_models_df.columns:
        print("GUI Warning: Global models file missing or malformed.")
        return pd.DataFrame(
            [["Global models config error", "Unable to load models"]],
            columns=["Worker", "Status"],
        )

    global_models = set(global_models_df["Model"].dropna().astype(str).str.strip())

    # Step 3: For each worker, skip disabled or unhealthy to avoid timeouts; otherwise fetch available models
    for _, row in workers_df.iterrows():
        worker_name = row.get("name")
        worker_url = row.get("url")
        enabled = _coerce_bool(row.get("enabled", True), True)
        healthy = _coerce_bool(row.get("healthy", True), True)

        if not enabled:
            worker_status_list.append([worker_name, "Disabled", ""])
            continue
        if not healthy:
            worker_status_list.append([worker_name, "Offline", ""])
            continue

        api_tags_url = f"{str(worker_url).rstrip('/')}/api/tags"
        try:
            resp = requests.get(api_tags_url, timeout=5)
            if resp.status_code != 200:
                status_msg = f"Fetch Error: HTTP {resp.status_code}"
                worker_status_list.append([worker_name, status_msg, ""])
                continue

            tags_data = resp.json()
            if isinstance(tags_data, dict) and "models" in tags_data:
                worker_models = set(
                    str(m.get("name", "")).strip() for m in tags_data["models"]
                )
            else:
                worker_status_list.append(
                    [worker_name, "Fetch Error: Unexpected response format", ""]
                )
                continue

            missing_models = global_models - worker_models
            extra_models = worker_models - global_models

            status = (
                "None" if not missing_models else f"{', '.join(sorted(missing_models))}"
            )
            extra_str = ", ".join(sorted(extra_models)) if extra_models else ""

            worker_status_list.append([worker_name, status, extra_str])

        except Exception as e:
            worker_status_list.append([worker_name, f"Fetch Error: {str(e)}", ""])

    return pd.DataFrame(worker_status_list, columns=["Worker", "Missing", "Additional"])


def get_users():
    # create if not exists. Structure: user,expirationDate,accessKey
    if not os.path.exists(AUTHORIZED_USERS_CONFIG_PATH):
        print("GUI Warning: Authorized users config file not found. Creating new file.")
        pd.DataFrame(columns=["user", "expirationDate", "accessKey"]).to_csv(
            AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding="utf-8"
        )
    return pd.read_csv(AUTHORIZED_USERS_CONFIG_PATH)


def get_users_markdown():
    users_df = get_users()
    if users_df.empty or "user" not in users_df.columns:
        return "No registered users."
    users_sorted = users_df.sort_values(by="user", key=lambda x: x.str.lower())
    header = "| Username | Registered until | AccessKey |\n|---|---|---|"
    rows = [
        f"| **{row['user']}** | {row['expirationDate']} | `{row['accessKey']}` |"
        for _, row in users_sorted.iterrows()
    ]
    return (
        "### Registered Users\n\n" + header + "\n" + "\n".join(rows)
        if rows
        else "No registered users."
    )


def add_global_model(model_name):
    # if model name contains :latest, show warning and skip
    if ":latest" in model_name.lower():
        gr.Warning(
            f"  - Skipping {model_name}: 'latest' tag not allowed. Please specify actual size so that we can manage model sizes properly."
        )
        return pd.read_csv(MODELS_CONFIG_PATH)
    # Parse and validate size (accept decimals and case-insensitive units)
    parsed_size_b = _parse_model_size_from_string(model_name)
    if parsed_size_b is None:
        gr.Warning(
            f"  - Skipping {model_name}: Please specify model size like ':4b' or ':500m'."
        )
        return pd.read_csv(MODELS_CONFIG_PATH)

    # Persist size for backend routing heuristics
    try:
        _save_model_size(model_name, float(parsed_size_b))
    except Exception:
        pass

    required_vram = _estimate_required_vram_mb(parsed_size_b)

    # Select workers: enabled + healthy with sufficient VRAM; else largest VRAM single worker
    try:
        workers_df = pd.read_csv(WORKER_CONFIG_PATH)
    except Exception:
        workers_df = pd.DataFrame(columns=["name", "url", "enabled", "healthy"])

    if "healthy" not in workers_df.columns:
        workers_df["healthy"] = True

    candidates = []  # (name, url, total_vram)
    for _, row in workers_df.iterrows():
        name = row.get("name")
        url = row.get("url")
        enabled = _coerce_bool(row.get("enabled", True), True)
        healthy = _coerce_bool(row.get("healthy", True), True)
        if not (enabled and healthy):
            continue
        total_vram = _get_worker_vram_total_mb(url, name, workers_df)
        candidates.append((name, url, total_vram))

    if not candidates:
        gr.Warning("No enabled and healthy workers available to pull the model.")
        return pd.read_csv(MODELS_CONFIG_PATH)

    fitting = [
        (n, u, v)
        for (n, u, v) in candidates
        if isinstance(v, (int, float)) and v is not None and v >= required_vram
    ]

    if fitting:
        chosen = fitting
        gr.Info(
            "GPU-fit check: pulling on workers where the model fits in VRAM: "
            + ", ".join([f"{n}({int(v)}MB)" for n, _, v in chosen])
        )
    else:
        known = [
            c
            for c in candidates
            if isinstance(c[2], (int, float)) and c[2] is not None and c[2] > 0
        ]
        if known:
            best = max(known, key=lambda x: x[2])
            chosen = [best]
            gr.Info(
                f"GPU-fit check: no worker can fully host on GPU; pulling only on largest VRAM worker {best[0]} ({int(best[2])}MB)."
            )
        else:
            chosen = [candidates[0]]
            gr.Warning(
                "GPU-fit check: VRAM totals unknown. Pulling only on the first enabled, healthy worker as a fallback."
            )

    # Perform pulls only on chosen workers
    for worker_name, url, _ in chosen:
        if not url:
            continue
        gr.Info(f"Trying to add model {model_name} to {worker_name}")

        last_reported_percent = -1
        try:
            response = requests.post(
                url=f"{url}/api/pull",
                json={"model": model_name, "stream": True},
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))

                    if data.get("error"):
                        gr.Warning(f"  - Failed to pull {model_name}: {data['error']}")

                    elif "pulling manifest" in data.get("status", ""):
                        gr.Info(f"  - Pulling {model_name}")
                    elif "total" in data and "completed" in data.keys():
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total > 0:
                            percent_done = (completed / total) * 100
                            if percent_done - last_reported_percent >= 25 or (
                                percent_done >= 99 and last_reported_percent < 99
                            ):
                                last_reported_percent = percent_done
                                gr.Info(f"  - Downloading... {percent_done:.1f}%")

                    if data.get("status") == "success":
                        gr.Info(
                            f"  + {model_name} pulled successfully to {worker_name}."
                        )

                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        except requests.exceptions.RequestException as e:
            gr.Info(f"  - EXCEPTION for {model_name}: {e}")

    gr.Info("Update process finished.")

    new_model = pd.DataFrame(
        [{"Model": model_name, "LastUsed": datetime.today().strftime("%d.%m.%Y")}]
    )
    models = pd.concat([pd.read_csv(MODELS_CONFIG_PATH), new_model], ignore_index=True)

    models.to_csv(MODELS_CONFIG_PATH, index=False, encoding="utf-8")
    return get_global_models()


def add_user(new_user_name, new_user_expiration=None):
    def generate_key(length=20):
        """Generate a random key of given length"""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789"
        return "".join(random.choice(chars) for _ in range(length))

    if len(new_user_name) == 0 or new_user_name == "":
        gr.Warning("No username provided.")
        return get_users_markdown()

    if not new_user_expiration:
        new_user_expiration = (datetime.today() + relativedelta(months=6)).strftime(
            "%d.%m.%Y"
        )

    new_user = pd.DataFrame(
        [
            {
                "user": new_user_name,
                "expirationDate": new_user_expiration,
                "accessKey": generate_key(),
            }
        ]
    )
    users = pd.concat(
        [pd.read_csv(AUTHORIZED_USERS_CONFIG_PATH), new_user], ignore_index=True
    )

    users.to_csv(AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding="utf-8")
    return get_users_markdown()


def add_worker(new_worker_name, new_worker_url):
    if new_worker_url == "":
        return get_worker_status()

    workers = pd.read_csv(WORKER_CONFIG_PATH)

    # Ensure healthy column exists
    if "healthy" not in workers.columns:
        workers["healthy"] = True

    new_worker = pd.DataFrame(
        [
            {
                "name": new_worker_name,
                "url": new_worker_url,
                "enabled": True,
                "healthy": True,
            }
        ]
    )
    workers = pd.concat([workers, new_worker], ignore_index=True)
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding="utf-8")

    return get_worker_status()


def add_missing_models_generator():
    logs = []
    # Snapshot workers and their VRAM totals for decisions
    try:
        workers_df = pd.read_csv(WORKER_CONFIG_PATH)
    except Exception:
        workers_df = pd.DataFrame(columns=["name", "url", "enabled", "healthy"])

    if "healthy" not in workers_df.columns:
        workers_df["healthy"] = True

    # Build quick lookup by worker name
    worker_rows = {str(row.get("name")): row for _, row in workers_df.iterrows()}

    server_status = get_worker_status()
    worker_models = get_worker_models()

    for worker in worker_models.itertuples(index=False):
        status = getattr(worker, "Missing")
        if not status or status == "None":
            continue

        worker_name = getattr(worker, "Worker")
        # use config row to respect healthy/enabled and get URL
        row = worker_rows.get(worker_name)
        if not row is None:
            enabled = _coerce_bool(row.get("enabled", True), True)
            healthy = _coerce_bool(row.get("healthy", True), True)
            url = row.get("url")
        else:
            enabled, healthy, url = (
                True,
                True,
                next(
                    (entry[1] for entry in server_status if entry[0] == worker_name), ""
                ),
            )

        if not enabled:
            logs.append(f"[Skipping {worker_name}] :: Worker disabled")
            yield "\n".join(logs)
            continue
        if not healthy:
            logs.append(f"[Skipping {worker_name}] :: Worker offline")
            yield "\n".join(logs)
            continue

        if not url:
            logs.append(f"[Skipping {worker_name}] :: URL not found")
            yield "\n".join(logs)
            continue

        logs.append(f"[Updating {worker_name}]")
        yield "\n".join(logs)

        models = [model.strip() for model in status.split(",") if model.strip()]

        for model_name in models:
            # Respect VRAM fit before pulling
            size_b = _parse_model_size_from_string(model_name)
            if size_b is None:
                logs.append(
                    f"  - Skipping {model_name}: missing or invalid size tag; expected like ':7b' or ':500m'"
                )
                yield "\n".join(logs)
                continue
            required_vram = _estimate_required_vram_mb(size_b)
            total_vram = _get_worker_vram_total_mb(url, worker_name, workers_df)
            if (
                isinstance(total_vram, (int, float))
                and total_vram is not None
                and total_vram < required_vram
            ):
                logs.append(
                    f"  - Skipping {model_name}: requires ~{int(required_vram)}MB VRAM, {worker_name} has {int(total_vram)}MB"
                )
                yield "\n".join(logs)
                continue

            last_reported_percent = -1
            try:
                response = requests.post(
                    url=f"{url}/api/pull",
                    json={"model": model_name, "stream": True},
                    stream=True,
                    timeout=300,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        log_line = ""
                        if data.get("error"):
                            log_line = (
                                f"  - Failed to pull {model_name}: {data['error']}"
                            )
                        elif "pulling manifest" in data.get("status", ""):
                            log_line = f"  - Pulling {model_name}:"
                        elif "total" in data and "completed" in data.keys():
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            if total > 0:
                                percent_done = (completed / total) * 100
                                if percent_done - last_reported_percent >= 10 or (
                                    percent_done >= 99 and last_reported_percent < 99
                                ):
                                    last_reported_percent = percent_done
                                    log_line = f"  - Downloading... {percent_done:.1f}%"
                        if log_line and logs[-1] != log_line:
                            logs.append(log_line)
                            yield "\n".join(logs)

                        if data.get("status") == "success":
                            logs.append(
                                f"  + {model_name} pulled successfully to {worker_name}."
                            )
                            yield "\n".join(logs)
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

            except requests.exceptions.RequestException as e:
                logs.append(f"  - EXCEPTION for {model_name}: {e}")
                yield "\n".join(logs)

    logs.append("\nUpdate process finished.")
    yield "\n".join(logs)


def remove_user(user_name):
    users = pd.read_csv(AUTHORIZED_USERS_CONFIG_PATH)
    users = users[users["user"] != user_name]
    users.to_csv(AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding="utf-8")
    return get_users_markdown()


def remove_model(model_name):
    models = pd.read_csv(MODELS_CONFIG_PATH)

    models = models[models["Model"] != model_name]
    models.to_csv(MODELS_CONFIG_PATH, index=False, encoding="utf-8")

    updated_models = get_global_models()

    logs = []
    server_status = get_worker_status()

    logs.append(f"Removing: {model_name}")
    yield "\n".join(logs), updated_models

    for worker_name, url, *_ in server_status:
        try:
            response = requests.delete(
                url=f"{url}/api/delete", json={"name": model_name}, timeout=120
            )

            if response.status_code == 404:
                logs.append(
                    f"  - Model '{model_name}' not found on {worker_name} (already removed)."
                )
                yield "\n".join(logs), updated_models
                continue

            response.raise_for_status()
            logs.append(f"  + Successfully removed '{model_name}' from {worker_name}.")
            yield "\n".join(logs), updated_models

        except requests.exceptions.RequestException as e:
            logs.append(f"  - EXCEPTION for {worker_name}: {e}")
            yield "\n".join(logs), updated_models

    logs.append("\nRemoval process finished.")
    yield "\n".join(logs), updated_models


def remove_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers = workers[workers["name"] != worker_name]
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding="utf-8")

    return get_worker_status()


def disable_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers.loc[workers["name"] == worker_name, "enabled"] = False
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding="utf-8")

    return get_worker_status()


def enable_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers.loc[workers["name"] == worker_name, "enabled"] = True
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding="utf-8")

    return get_worker_status()


def login(password):
    """
    Checks the password. If correct, it returns updates to make all admin components
    visible (also hides login field+button). Otherwise, it returns an error message.
    """
    if password == CORRECT_PASSWORD:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="Login successful! Admin controls enabled.", visible=True),
        )
    else:
        # Return 8 outputs matching the wired components, with only the status message changed.
        return (
            gr.update(),  # worker_model_management (no change)
            gr.update(),  # user_management (no change)
            gr.update(),  # worker_mangagement (no change)
            gr.update(),  # admin_logs (no change)
            gr.update(),  # admin_stats (no change)
            gr.update(),  # admin_login_control (no change)
            gr.update(),  # model_management (no change)
            gr.update(value="Incorrect password.", visible=True),  # login_status
        )


def _rewrite_url_to_helper_port(worker_url):
    if not isinstance(worker_url, str) or not re.match(r"^https?://", worker_url):
        return None
    if "://" in worker_url:
        proto, rest = worker_url.split("://", 1)
        if "/" in rest:
            host_part, path_part = rest.split("/", 1)
        else:
            host_part = rest
            path_part = ""
        host_part = re.sub(r":\d+", ":18034", host_part)
        if ":" not in host_part.split("/")[-1]:
            host_part = f"{host_part}:18034"
        if path_part:
            new_url = f"{proto}://{host_part}/{path_part}"
        else:
            new_url = f"{proto}://{host_part}"
    else:
        new_url = worker_url
    return new_url


def update_ollama(worker_status):
    """
    Calls the /ollama/update endpoint on each enabled worker and displays results.
    Uses port 18034 for the worker application.
    """
    logs = []
    for worker in worker_status.itertuples(index=False):
        worker_name = worker[0]
        worker_url = worker[1]
        enabled = worker[-1]
        if not enabled or str(enabled).lower() == "false":
            logs.append(f"[{worker_name}] Skipped (disabled)")
            continue

        new_url = _rewrite_url_to_helper_port(worker_url)
        if new_url is None:
            logs.append(
                f"[{worker_name}] Skipped: Invalid URL '{worker_url}' (no scheme supplied)"
            )
            continue

        try:
            update_url = f"{new_url.rstrip('/')}/ollama/update"
            response = requests.post(
                update_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=10
            )
            if response.status_code == 202:
                msg = response.json().get("message", "Update started.")
                logs.append(f"[{worker_name}] Update triggered: {msg}")
            else:
                logs.append(
                    f"[{worker_name}] Error: HTTP {response.status_code} - {response.text}"
                )
        except Exception as e:
            logs.append(f"[{worker_name}] Exception: {e}")
    gr.Info("\n".join(logs))
    return get_worker_status()


def update_workers_git(worker_status):
    """
    Calls the /worker/trigger-update endpoint on each enabled worker to trigger
    a git-based self-update. Uses port 18034 for the worker application.
    """
    logs = []
    for worker in worker_status.itertuples(index=False):
        worker_name = worker[0]
        worker_url = worker[1]
        enabled = worker[-1]
        if not enabled or str(enabled).lower() == "false":
            logs.append(f"[{worker_name}] Skipped (disabled)")
            continue

        new_url = _rewrite_url_to_helper_port(worker_url)
        if new_url is None:
            logs.append(
                f"[{worker_name}] Skipped: Invalid URL '{worker_url}' (no scheme supplied)"
            )
            continue

        try:
            update_url = f"{new_url.rstrip('/')}/worker/trigger-update"
            response = requests.post(
                update_url,
                headers={"x-api-key": OLLAMA_HELPER_API_KEY},
                timeout=10,
            )
            if response.status_code == 202:
                msg = response.json().get("message", "Git update triggered.")
                logs.append(f"[{worker_name}] {msg}")
            elif response.status_code == 409:
                logs.append(f"[{worker_name}] Update already in progress.")
            else:
                logs.append(
                    f"[{worker_name}] Error: HTTP {response.status_code} - {response.text}"
                )
        except Exception as e:
            logs.append(f"[{worker_name}] Exception: {e}")
    gr.Info("\n".join(logs))
    return get_worker_status()


def update_workers_tar(worker_status, tar_file):
    """
    Uploads a .tar.gz archive to each enabled worker via the /worker/self-update
    endpoint. Uses port 18034 for the worker application.
    """
    if tar_file is None:
        gr.Warning("No archive file provided.")
        return get_worker_status()

    if not tar_file.endswith((".tar.gz", ".tgz")):
        gr.Warning("Uploaded file must be a .tar.gz archive.")
        return get_worker_status()

    logs = []
    with open(tar_file, "rb") as f:
        archive_data = f.read()

    filename = os.path.basename(tar_file)
    for worker in worker_status.itertuples(index=False):
        worker_name = worker[0]
        worker_url = worker[1]
        enabled = worker[-1]
        if not enabled or str(enabled).lower() == "false":
            logs.append(f"[{worker_name}] Skipped (disabled)")
            continue

        new_url = _rewrite_url_to_helper_port(worker_url)
        if new_url is None:
            logs.append(
                f"[{worker_name}] Skipped: Invalid URL '{worker_url}' (no scheme supplied)"
            )
            continue

        try:
            update_url = f"{new_url.rstrip('/')}/worker/self-update"
            response = requests.post(
                update_url,
                headers={"x-api-key": OLLAMA_HELPER_API_KEY},
                files={"file": (filename, archive_data, "application/gzip")},
                timeout=30,
            )
            if response.status_code == 202:
                msg = response.json().get("message", "Update started.")
                logs.append(f"[{worker_name}] Update triggered: {msg}")
            elif response.status_code == 409:
                logs.append(f"[{worker_name}] Update already in progress.")
            else:
                logs.append(
                    f"[{worker_name}] Error: HTTP {response.status_code} - {response.text}"
                )
        except Exception as e:
            logs.append(f"[{worker_name}] Exception: {e}")
    gr.Info("\n".join(logs))
    return get_worker_status()


def clean_expired_users():
    today = datetime.today().date()
    temp_file = NamedTemporaryFile(mode="w", delete=False, newline="")
    get_users()

    try:
        with (
            open(AUTHORIZED_USERS_CONFIG_PATH, mode="r", newline="") as csvfile,
            temp_file,
        ):
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames or []
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()

            for rec in reader:
                try:
                    row_dict = {k: rec[k] for k in (fieldnames or rec.keys())}
                    exp_str = str(row_dict.get("expirationDate", "")).strip()
                    exp_date = datetime.strptime(exp_str, "%d.%m.%Y").date()
                    if exp_date >= today:
                        writer.writerow(row_dict)
                except Exception as e:
                    print(f"Skipping row due to error: {e}")

        os.replace(temp_file.name, AUTHORIZED_USERS_CONFIG_PATH)
    except Exception:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
        raise


def start_cleanup_thread(intervall_hours=2):
    sleep_time_seconds = intervall_hours * 60 * 60

    def loop():
        while True:
            clean_expired_users()
            time.sleep(sleep_time_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def usage_logger_background():
    """Background thread to periodically flush in-memory usage logs if needed."""
    # This is a placeholder for future batching if needed.
    pass


def start_usage_logger_thread():
    thread = threading.Thread(target=usage_logger_background, daemon=True)
    thread.start()
    return thread


def load_usage_stats():
    if not os.path.exists(USAGE_STATS_PATH):
        return []
    try:
        with open(USAGE_STATS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading usage stats: {e}")
        return []


def aggregate_usage_stats(window_hours=None, window_days=None, window_months=None):
    """Aggregate usage stats for a given time window."""
    stats = load_usage_stats()
    now = datetime.utcnow()
    if window_hours:
        cutoff = now - pd.Timedelta(hours=window_hours)
    elif window_days:
        cutoff = now - pd.Timedelta(days=window_days)
    elif window_months:
        cutoff = now - pd.DateOffset(months=window_months)
    else:
        cutoff = datetime.min
    per_user = {}
    for entry in stats:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts < cutoff:
                continue
            user = entry.get("user", "unknown")
            duration = float(entry.get("duration", 0))
            if user not in per_user:
                per_user[user] = {"calls": 0, "duration": 0.0}
            per_user[user]["calls"] += 1
            per_user[user]["duration"] += duration
        except Exception:
            continue
    return per_user


def plot_usage(window_label, window_hours=None, window_days=None, window_months=None):
    per_user = aggregate_usage_stats(window_hours, window_days, window_months)
    users = list(per_user.keys())
    calls = [per_user[u]["calls"] for u in users]
    durations = [per_user[u]["duration"] / 60.0 for u in users]  # convert to minutes

    fig_calls = go.Figure([go.Bar(x=users, y=calls)])
    fig_calls.update_layout(
        title=f"Calls per User ({window_label})",
        xaxis_title="User",
        yaxis_title="Calls",
    )

    fig_durations = go.Figure([go.Bar(x=users, y=durations)])
    fig_durations.update_layout(
        title=f"Total Usage Time per User ({window_label})",
        xaxis_title="User",
        yaxis_title="Minutes",
    )

    return fig_calls, fig_durations


def get_all_usage_plots():
    plots = []
    for label, kwargs in [
        ("Last 24 Hours", {"window_hours": 24}),
        ("Last 7 Days", {"window_days": 7}),
        ("Last 12 Months", {"window_months": 12}),
    ]:
        fig_calls, fig_durations = plot_usage(label, **kwargs)
        plots.append(fig_calls)
        plots.append(fig_durations)
    return plots


def create_gui():
    """
    Creates the Gradio interface.
    """
    default_log_lines = 50

    with gr.Blocks(title="Ollama Proxy Server") as demo:
        gr.Markdown("# Ollama Proxy Server")

        with gr.Tabs():
            """
                <----- MODELS AND WORKERS ----->
            """
            with gr.TabItem(label="Workers"):
                gr.Markdown("## Configured Ollama Workers")
                worker_status = gr.DataFrame(
                    headers=[
                        "Name",
                        "URL",
                        "Running Models",
                        "GPU Usage",
                        "GPU VRAM Usage",
                        "Ollama Version",
                        "Activity",
                        "Enabled",
                    ],
                    interactive=False,
                    row_count=(10, "dynamic"),
                )
                with gr.Row(visible=False) as worker_mangagement:
                    with gr.Column(scale=1):
                        with gr.Row():
                            new_worker_name = gr.Textbox(
                                label="Worker Name", type="text"
                            )
                            new_worker_url = gr.Textbox(label="Worker URL", type="text")
                        with gr.Row():
                            with gr.Column(scale=1):
                                add_worker_btn = gr.Button("Add Worker")
                                remove_worker_btn = gr.Button("Remove Worker")
                            with gr.Column(scale=1):
                                enable_worker_btn = gr.Button("Enable")
                                disable_worker_btn = gr.Button("Disable")
                        update_ollama_btn = gr.Button("Update Ollama Version")
                        update_workers_git_btn = gr.Button("Git Pull Update Workers")
                    with gr.Column(scale=1):
                        gr.Textbox(
                            label="Ollama Update Log",
                            lines=10,
                            max_lines=10,
                            interactive=False,
                            show_copy_button=True,
                        )
                        worker_tar_file = gr.File(
                            label="Worker Update Archive (.tar.gz)",
                            file_types=[".tar.gz", ".tgz"],
                            type="filepath",
                        )
                        update_workers_tar_btn = gr.Button(
                            "Upload Worker Update (Tar Archive)"
                        )

                gr.Markdown("<br>")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Global Models")

                        global_models = gr.DataFrame(
                            headers=["Model", "Last Used"],
                            interactive=False,
                            row_count=(5, "dynamic"),
                            col_count=2,
                        )

                        model_name = gr.Textbox(label="Model", type="text")
                        add_model_btn = gr.Button("Add Model")
                        with gr.Row(visible=False) as model_management:
                            remove_model_btn = gr.Button("Remove Model")

                    with gr.Column(scale=1, visible=False) as worker_model_management:
                        gr.Markdown("## Available Models")

                        worker_models = gr.DataFrame(
                            headers=["Worker", "Status", "Additional models"],
                            interactive=False,
                            row_count=(10, "dynamic"),
                        )

                        pull_missing_btn = gr.Button("Pull missing models")

                        update_logs = gr.Textbox(
                            label="Log",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True,
                        )

            """
                <----- USERS ----->
            """
            with gr.TabItem(label="Users", visible=False) as user_management:
                users_markdown = gr.Markdown(
                    get_users_markdown, elem_id="users-markdown"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        new_user_name = gr.Textbox(label="User", type="text")
                        new_user_expiration = gr.Textbox(
                            label="Optional: Expiration Date (Default: 6 Months)",
                            type="text",
                        )
                    with gr.Column(scale=1):
                        add_user_btn = gr.Button("Add User")
                        remove_user_btn = gr.Button("Remove User")

            """
                <----- LOGS ----->
            """
            with gr.TabItem(label="Logs", visible=False) as admin_logs:
                gr.Markdown("## Access Log Viewer")

                log_output_textbox = gr.Textbox(
                    label="Log Entries",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True,
                )

                with gr.Row():
                    log_lines_input = gr.Number(
                        value=default_log_lines,
                        label="Number of log lines",
                        minimum=1,
                        step=1,
                        precision=0,
                    )
                    refresh_logs_btn = gr.Button("Refresh Logs")

                def update_logs_display(lines_to_show_from_input):
                    return get_logs(num_lines=int(lines_to_show_from_input))

            with gr.TabItem(label="Usage Statistics", visible=False) as admin_stats:
                gr.Markdown("## Usage Statistics (per user)")

                usage_plot_24h_calls = gr.Plot(label="Calls per User (24h)")
                usage_plot_24h_time = gr.Plot(label="Total Usage Time (24h)")
                usage_plot_7d_calls = gr.Plot(label="Calls per User (7d)")
                usage_plot_7d_time = gr.Plot(label="Total Usage Time (7d)")
                usage_plot_12mo_calls = gr.Plot(label="Calls per User (12mo)")
                usage_plot_12mo_time = gr.Plot(label="Total Usage Time (12mo)")
                refresh_usage_btn = gr.Button("Refresh Usage Stats")

                def refresh_usage_plots():
                    figs = get_all_usage_plots()
                    # figs: [24h_calls, 24h_time, 7d_calls, 7d_time, 12mo_calls, 12mo_time]
                    return figs

                refresh_usage_btn.click(
                    fn=refresh_usage_plots,
                    inputs=None,
                    outputs=[
                        usage_plot_24h_calls,
                        usage_plot_24h_time,
                        usage_plot_7d_calls,
                        usage_plot_7d_time,
                        usage_plot_12mo_calls,
                        usage_plot_12mo_time,
                    ],
                )

        """
            <----- ADMIN LOGIN ----->
        """
        gr.Markdown("---")
        gr.Markdown("#### Admin Login")
        with gr.Row() as admin_login_control:
            password_input = gr.Textbox(
                label="Password", type="password", placeholder="Enter admin password..."
            )
            login_button = gr.Button("Login")
        login_status = gr.Markdown(visible=False)

        # ------> events <------
        password_input.submit(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management,
                user_management,
                worker_mangagement,
                admin_logs,
                admin_stats,
                admin_login_control,
                model_management,
                login_status,
            ],
        )

        login_button.click(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management,
                user_management,
                worker_mangagement,
                admin_logs,
                admin_stats,
                admin_login_control,
                model_management,
                login_status,
            ],
        )

        add_user_btn.click(
            fn=lambda name, expiration: add_user(name, new_user_expiration=expiration),
            inputs=[new_user_name, new_user_expiration],
            outputs=users_markdown,
        )

        remove_user_btn.click(
            fn=lambda name: remove_user(name),
            inputs=[new_user_name],
            outputs=users_markdown,
        )

        add_worker_btn.click(
            fn=lambda new_worker_name, new_worker_url: add_worker(
                new_worker_name, new_worker_url
            ),
            inputs=[new_worker_name, new_worker_url],
            outputs=worker_status,
        )

        add_model_btn.click(
            fn=lambda model_name: add_global_model(model_name),
            inputs=[model_name],
            outputs=global_models,
        )

        disable_worker_btn.click(
            fn=lambda edit_worker_name: disable_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status,
        )

        enable_worker_btn.click(
            fn=lambda edit_worker_name: enable_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status,
        )

        remove_worker_btn.click(
            fn=lambda edit_worker_name: remove_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status,
        )

        update_ollama_btn.click(
            fn=update_ollama, inputs=[worker_status], outputs=worker_status
        )

        update_workers_git_btn.click(
            fn=update_workers_git, inputs=[worker_status], outputs=worker_status
        )

        update_workers_tar_btn.click(
            fn=update_workers_tar,
            inputs=[worker_status, worker_tar_file],
            outputs=worker_status,
        )

        pull_missing_btn.click(
            fn=add_missing_models_generator, inputs=None, outputs=update_logs
        )

        remove_model_btn.click(
            fn=remove_model, inputs=[model_name], outputs=[update_logs, global_models]
        )

        refresh_logs_btn.click(
            fn=update_logs_display, inputs=[log_lines_input], outputs=log_output_textbox
        )

        # Initial loads
        def on_load():
            server_status = get_worker_status()
            logs = get_logs(num_lines=default_log_lines)
            models = get_global_models()
            worker_status = get_worker_models()
            users_md = get_users_markdown()
            usage_figs = get_all_usage_plots()
            return (server_status, logs, models, worker_status, users_md, *usage_figs)

        demo.load(
            on_load,
            inputs=None,
            outputs=[
                worker_status,
                log_output_textbox,
                global_models,
                worker_models,
                users_markdown,
                usage_plot_24h_calls,
                usage_plot_24h_time,
                usage_plot_7d_calls,
                usage_plot_7d_time,
                usage_plot_12mo_calls,
                usage_plot_12mo_time,
            ],
        )

    return demo


def start_gui(gui_port_to_use):
    """
    Launches the Gradio GUI.
    Passes the server config getter to create_gui.

    launch command: python proxy.py --config ../workers.csv --users_list ../authorized_users.csv --log_path access_log.txt --port 8000 --gui_port 7860 --model ../models.txt
    """
    print(f"GUI: Attempting to launch Gradio GUI on port {gui_port_to_use}...")
    gui_app = create_gui()
    try:
        gui_app.launch(
            server_name="0.0.0.0",
            server_port=int(gui_port_to_use),
            share=False,
            show_api=False,
        )
        print(f"GUI: Gradio GUI is running on http://localhost:{gui_port_to_use}")
    except Exception as e:
        print(f"GUI Error: Failed to launch Gradio GUI: {e}")


def start_proxy_server(port, request_handler_class):
    """Starts the HTTP proxy server."""
    try:
        proxy_server = ThreadedHTTPServer(("", port), request_handler_class)
        print(f"Running Ollama proxy server on port {port}")
        proxy_server.serve_forever()
    except OSError as e:
        print(f"Could not start proxy server on port {port}: {e}")
        print("The port might be already in use.")
    except Exception as e:
        print(f"An unexpected error occurred while starting the proxy server: {e}")


def main():
    global WORKER_CONFIG_PATH, AUTHORIZED_USERS_CONFIG_PATH, MODELS_PATH

    parser = argparse.ArgumentParser(
        description="Ollama Proxy Server with Security and Load Balancing"
    )
    parser.add_argument(
        "--config",
        default="workers.csv",
        help="Path to the server configuration file (default: workers.csv)",
    )
    parser.add_argument(
        "--log_path",
        default="access_log.txt",
        help="Path to the access log file (default: access_log.txt)",
    )
    parser.add_argument(
        "--users_list",
        default="authorized_users.csv",
        help="Path to the authorized users list file (default: authorized_users.csv)",
    )
    parser.add_argument(
        "--models",
        default="models.csv",
        help="Models available on all workers (default: models.csv)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port number for the proxy server (default: 8000)",
    )
    parser.add_argument(
        "--gui_port",
        type=int,
        default=7860,
        help="Port number for the Gradio GUI (default: 7860)",
    )
    args = parser.parse_args()

    WORKER_CONFIG_PATH = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    )
    AUTHORIZED_USERS_CONFIG_PATH = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.users_list)
    )
    LOG_FILE_PATH = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_path)
    )
    MODELS_PATH = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.models)
    )

    if not os.path.exists(WORKER_CONFIG_PATH):
        print("GUI Warning: Workers config file not found. Creating new file.")
        pd.DataFrame(columns=["name", "url", "enabled", "healthy"]).to_csv(
            WORKER_CONFIG_PATH, index=False, encoding="utf-8"
        )
    else:
        ensure_workers_csv_has_healthy_column()
    get_users()
    get_global_models()
    get_logs(1)

    print("Ollama Proxy server")
    print(f"Configuration file: {WORKER_CONFIG_PATH}")
    print(f"Users list file: {AUTHORIZED_USERS_CONFIG_PATH}")
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"Models file: {MODELS_PATH}")

    # Pre-warm worker cache (VRAM, available & loaded models) to avoid cold-start misses
    init_worker_state_cache()

    # Start health monitor thread to auto-toggle routing based on availability (faster reaction)
    start_health_monitor_thread(interval_seconds=15)

    # Start background refresher to keep worker cache (VRAM/models) warm
    start_worker_state_refresher_thread(interval_seconds=15, models_ttl_seconds=30)

    proxy_thread = threading.Thread(
        target=start_proxy_server, args=(args.port, RequestHandler), daemon=True
    )
    proxy_thread.start()

    start_cleanup_thread(intervall_hours=2)
    start_usage_logger_thread()
    start_gui(args.gui_port)

    if proxy_thread.is_alive():
        try:
            while proxy_thread.is_alive():
                proxy_thread.join(timeout=1)
        except KeyboardInterrupt:
            print("\nShutdown signal received. Exiting.")
        finally:
            print("Ollama Proxy Server shut down.")


if __name__ == "__main__":
    main()

"""
    TODO List
    1. complete update_ollama -> fastAPI config
    2¡. test using VM (SCS-AI-PROXY)
"""
