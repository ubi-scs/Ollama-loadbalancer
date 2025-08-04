import configparser
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from queue import Queue
import requests
import argparse
from pathlib import Path
import csv
import datetime
import threading

SERVERS_CONFIG = []
AUTHORIZED_USERS = {}
CONFIG_FILE_PATH = "config.info"
USERS_FILE_PATH = "authorized_users.txt"
LOG_FILE_PATH = "access_log.txt"
csv_file = "/Users/vincent/Documents/UNI/Arbeit/Hendric/OllamaProject/models.csv"

try:
    from gui import launch_gui
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"Could not import GUI module: {e}. GUI will not be available.")
    GUI_AVAILABLE = False
    def launch_gui(port):
        print("GUI module not found. GUI cannot be started.")


def get_config(filename):
    config = configparser.ConfigParser()
    if not Path(filename).exists():
        print(f"Config file {filename} not found. No servers will be loaded.")
        return []
    config.read(filename)
    parsed_servers = []
    for name in config.sections():
        try:
            parsed_servers.append((name, {'url': config[name]['url'], 'queue': Queue()}))
        except KeyError:
            print(f"Server entry '{name}' in {filename} is missing 'url'. Skipping.")
    return parsed_servers


def save_last_used(model):
    today = datetime.datetime.today().strftime("%d.%m.%Y")
    rows = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == model:
                row[1] = today
            rows.append(row)

    with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def get_authorized_users(filename):
    authorized_users = {}
    user_info_list = []

    if not Path(filename).exists():
        print(f"Authorized users file {filename} not found. No users will be loaded.")
        return authorized_users, user_info_list

    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("user") or not row.get("accessKey"):
                print(f"Missing 'User' or 'Access Key' in row: {row}")
                continue
            authorized_users[row["user"]] = row["accessKey"]
            user_info_list.append(row)

    return authorized_users, user_info_list


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
            if not auth_header or not auth_header.startswith('Bearer '):
                return False
            token = auth_header.split(' ')[1]
            user, key = token.split(':', 1)

            if AUTHORIZED_USERS.get(user) == key:
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
            if auth_header and auth_header.startswith('Bearer '):
                token_info = auth_header.split(' ')[1]
            self.add_access_log_entry(event='rejected', user=token_info, ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="Authentication failed")
            self.send_response(403)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Forbidden: Authentication failed"}).encode('utf-8'))
            return

        url = urlparse(self.path)
        path = url.path
        get_params = parse_qs(url.query) or {}

        post_data = b''
        if self.command == "POST":
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
            except (TypeError, ValueError):
                print("POST request without valid Content-Length.")
                pass


        if not SERVERS_CONFIG:
            print("No backend servers configured. Cannot proxy request.")
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Service Unavailable: No backend servers configured."}).encode('utf-8'))
            self.add_access_log_entry(event='error', user=self.user, ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="No backend servers")
            return

        min_queued_server = SERVERS_CONFIG[0]
        for server_entry in SERVERS_CONFIG:
            cs = server_entry[1]
            if cs['queue'].qsize() < min_queued_server[1]['queue'].qsize():
                min_queued_server = server_entry

        if path == '/api/generate' or path == '/api/chat' or path == '/v1/chat/completions':
            que = min_queued_server[1]['queue']
            self.add_access_log_entry(event="gen_request", user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize())
            que.put_nowait(1)
            try:
                post_data_dict = {}
                is_streaming = False
                if post_data:
                    try:
                        post_data_str = post_data.decode('utf-8')
                        post_data_dict = json.loads(post_data_str)

                        model = post_data_dict['model']
                        save_last_used(model)


                        is_streaming = post_data_dict.get("stream", False)
                    except (UnicodeDecodeError, json.JSONDecodeError) as json_err:
                        print(f"Could not parse POST data as JSON for {path}: {json_err}")

                response = requests.request(
                    self.command,
                    min_queued_server[1]['url'] + path,
                    params=get_params,
                    data=post_data,
                    headers={k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']}, # Forward relevant headers
                    stream=is_streaming
                )
                self._send_response(response)
            except requests.exceptions.RequestException as ex:
                print(f"Proxy request to {min_queued_server[0]} failed: {ex}")
                self.add_access_log_entry(event="gen_error",user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize(),error=str(ex))
                self.send_response(502) # Bad Gateway
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
            except Exception as ex_other: # Catch any other unexpected error
                print(f"Unexpected error during proxy to {min_queued_server[0]}: {ex_other}")
                self.add_access_log_entry(event="gen_error",user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize(),error=str(ex_other))
                self.send_response(500) # Internal Server Error
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))
            finally:
                if not que.empty(): # Ensure queue is not empty before get
                    que.get_nowait()
                else:
                    print(f"Attempted to get from an empty queue for server {min_queued_server[0]}. This might indicate a logic error.")
                self.add_access_log_entry(event="gen_done",user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize())
        else:
            try:
                response = requests.request(
                    self.command,
                    min_queued_server[1]['url'] + path,
                    params=get_params,
                    data=post_data, # Send raw bytes
                    headers={k: v for k, v in self.headers.items() if k.lower() not in ['host', 'connection', 'content-length']}
                )
                self._send_response(response)
            except requests.exceptions.RequestException as ex:
                print(f"Proxy request to {min_queued_server[0]} for non-gen endpoint failed: {ex}")
                self.send_response(502)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
            except Exception as ex_other: # Catch any other unexpected error
                print(f"Unexpected error during proxy (non-gen) to {min_queued_server[0]}: {ex_other}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error"}).encode('utf-8'))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

def start_proxy_server(port, request_handler_class):
    """Starts the HTTP proxy server."""
    try:
        proxy_server = ThreadedHTTPServer(('', port), request_handler_class)
        print(f'Running Ollama proxy server on port {port}')
        proxy_server.serve_forever()
    except OSError as e:
        print(f"Could not start proxy server on port {port}: {e}")
        print("The port might be already in use.")
    except Exception as e:
        print(f"An unexpected error occurred while starting the proxy server: {e}")


def main():
    global SERVERS_CONFIG, AUTHORIZED_USERS, CONFIG_FILE_PATH, USERS_FILE_PATH, LOG_FILE_PATH, DEACTIVATE_SECURITY

    parser = argparse.ArgumentParser(description="Ollama Proxy Server with Security and Load Balancing")
    parser.add_argument('--config', default="config.ini", help='Path to the server configuration file (default: config.ini)')
    parser.add_argument('--log_path', default="access_log.txt", help='Path to the access log file (default: access_log.txt)')
    parser.add_argument('--users_list', default="authorized_users.txt", help='Path to the authorized users list file (default: authorized_users.txt)')
    parser.add_argument('--models', default="models.txt", help='Models available on all workers (default: models.txt)')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the proxy server (default: 8000)')
    parser.add_argument('--gui_port', type=int, default=7860, help='Port number for the Gradio GUI (default: 7860)')
    parser.add_argument('-d', '--deactivate_security', action='store_true', help='Deactivates security layer (USE WITH CAUTION)')
    parser.add_argument('--no-gui', action='store_true', help='Do not launch the Gradio GUI')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.config
    USERS_FILE_PATH = args.users_list
    LOG_FILE_PATH = args.log_path
    DEACTIVATE_SECURITY = args.deactivate_security
    MODELS_FILE_PATH = args.models

    SERVERS_CONFIG = get_config(CONFIG_FILE_PATH)
    AUTHORIZED_USERS, _ = get_authorized_users(USERS_FILE_PATH)

    print("Ollama Proxy server")

    print(f"Configuration file: {CONFIG_FILE_PATH}")
    print(f"Users list file: {USERS_FILE_PATH}")
    print(f"Log file: {LOG_FILE_PATH}")

    proxy_thread = threading.Thread(target=start_proxy_server, args=(args.port, RequestHandler), daemon=True)
    proxy_thread.start()

    if not args.no_gui and GUI_AVAILABLE:

        launch_gui(args.gui_port,
                   SERVERS_CONFIG,
                   LOG_FILE_PATH,
                   MODELS_FILE_PATH,
                   USERS_FILE_PATH,
                   get_authorized_users)

    if (args.no_gui or not GUI_AVAILABLE) and proxy_thread.is_alive():
        try:
            while proxy_thread.is_alive():
                proxy_thread.join(timeout=1)
        except KeyboardInterrupt:
            print("\nShutdown signal received. Exiting.")
        finally:
            print("Ollama Proxy Server shut down.")


if __name__ == "__main__":
    main()