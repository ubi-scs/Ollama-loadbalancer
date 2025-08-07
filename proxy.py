import csv
import datetime
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests

USERS_FILE_PATH = "authorized_users.csv"
CONFIG_FILE_PATH = "workers.csv"
LOG_FILE_PATH = "access_log.txt"
MODELS_FILE_PATH = "models.csv"
DEACTIVATE_SECURITY = False


def get_user_key(user):
    users = pd.read_csv(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), USERS_FILE_PATH)))
    users = dict(zip(users['user'], users['accessKey']))
    return users.get(user, None)


def get_config():
    workers = pd.read_csv(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE_PATH)))

    worker_list = []
    for _,row in workers.iterrows():
        if row['enabled']:
            worker_list.append((row['name'], {'url': row['url'], 'queue': Queue()}))

    return worker_list if worker_list else None


def _save_last_used(model):
    today = datetime.datetime.today().strftime("%d.%m.%Y")
    rows = []
    models = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE_PATH))
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
            if not auth_header or not auth_header.startswith('Bearer '):
                return False
            token = auth_header.split(' ')[1]
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

        server_config = get_config()
        print(server_config)
        if not server_config:
            print("No backend servers configured. Cannot proxy request.")
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Service Unavailable: No backend servers configured."}).encode('utf-8'))
            self.add_access_log_entry(event='error', user=self.user, ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="No backend servers")
            return

        min_queued_server = server_config[0]
        for server_entry in server_config:
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
                        _save_last_used(model)

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
                traceback.print_exc()
                self.add_access_log_entry(event="gen_error",user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize(),error=str(ex))
                self.send_response(502) # Bad Gateway
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Gateway: Upstream server request failed"}).encode('utf-8'))
            except Exception as ex_other: # Catch any other unexpected error
                print(f"Unexpected error during proxy to {min_queued_server[0]}: {ex_other}")
                traceback.print_exc()
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