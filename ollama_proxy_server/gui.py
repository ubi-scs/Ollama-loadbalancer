import argparse
import configparser
import random
import threading
from queue import Queue
import gradio as gr
from pathlib import Path
import requests
import pandas as pd
import json
import os
import csv
from proxy import RequestHandler, ThreadedHTTPServer

CORRECT_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not CORRECT_PASSWORD:
    raise RuntimeError(f"ADMIN_PASSWORD environment variable not set. Please provide a secret Admin password.")

OLLAMA_HELPER_API_KEY = os.environ.get("OLLAMA_HELPER_API_KEY")
if not OLLAMA_HELPER_API_KEY:
    raise RuntimeError(f"OLLAMA_HELPER_API_KEY environment variable not set. Please provide a secret OLLAMA_HELPER_API_KEY.")


def generate_key(length=10):
    """Generate a random key of given length"""
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;,.<>?/~'
    return ''.join(random.choice(chars) for _ in range(length))


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
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="Login successful! Admin controls enabled.", visible=True)
        )
    else:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(value="Incorrect password.", visible=True)
        )


def get_worker_status(current_servers_config):
    status_data = []
    try:
        for name, details in current_servers_config:

            if isinstance(details, dict) and 'url' in details and 'queue' in details:
                server_url = details['url']
                queue_size = details['queue'].qsize() if hasattr(details['queue'], 'qsize') else 'N/A'

                running_models_str = "N/A"  # Default
                api_ps_url = f"{server_url.rstrip('/')}/api/ps"

                try:
                    response = requests.get(api_ps_url, timeout=5)
                    if response.status_code == 200:
                        ps_data = response.json()
                        if ps_data and "models" in ps_data and isinstance(ps_data["models"], list):
                            model_names = [model.get("name", "UnknownModel") for model in ps_data["models"]]
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
                    print(f"GUI: Error fetching /api/ps for server {name}: {e_ps}")

                try:
                    gpu_util_url = f"{server_url.replace('11434','8000')}/gpu/utilization"
                    response = requests.get(gpu_util_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=5)
                    if response.status_code == 200:
                        util_data = response.json()
                        gpu_utilization_str = f"{util_data.get('gpu_utilization_percent', 0.0):.1f}%"
                    else:
                        gpu_utilization_str = f"Err {response.status_code}"
                except Exception as e_util:
                    gpu_utilization_str = "Fetch Error"
                    print(f"GUI: Error fetching GPU utilization for server {name}: {e_util}")

                    # Fetch VRAM usage
                try:
                    vram_url = f"{server_url.replace('11434','8000')}/gpu/vram"
                    response = requests.get(vram_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=5)
                    if response.status_code == 200:
                        vram_data = response.json()
                        used = vram_data.get("vram_used_mb", 0)
                        total = vram_data.get("vram_total_mb", 1)
                        vram_usage_str = f"{int(used)}/{int(total)}MB"
                    else:
                        vram_usage_str = f"Err {response.status_code}"
                except Exception as e_vram:
                    vram_usage_str = "Fetch Error"
                    print(f"GUI: Error fetching VRAM info for server {name}: {e_vram}")

                    # Fetch version
                try:
                    api_version_url = f"{server_url.rstrip('/')}/api/version"
                    response = requests.get(api_version_url, timeout=5)
                    if response.status_code == 200:
                        version_data = response.json()
                        ollama_version = version_data.get("version", "JSON Error")
                    else:
                        ollama_version = f"Err {response.status_code}"
                except Exception as e_version:
                    ollama_version = "Fetch Error"
                    print(f"GUI: Error fetching Ollama version info for server {name}: {e_version}")

                status_data.append([name, server_url,
                                    queue_size,
                                    running_models_str,
                                    gpu_utilization_str,
                                    vram_usage_str,
                                    ollama_version])
            else:
                print(f"GUI Warning: Malformed server entry in config: '{name}': {details}")
                status_data.append([name, "Invalid Config", "N/A", "N/A"])

        if not status_data:
            print("GUI: current_servers_config was present, but status_data is empty. Returning default empty row.")
            return [["No valid server data found.", "", "", ""]]
        return status_data
    except Exception as e:
        print(f"GUI Error: Exception in get_server_status_for_gui: {e}")
        import traceback
        traceback.print_exc()
        return [["Error processing server data.", str(e), "", ""]]


def get_logs(log_file_path_str, num_lines=100):
    """
    Reads the last N lines from the access log file.
    """
    log_file_path = Path(log_file_path_str)
    if not log_file_path.exists():
        print(f"GUI Warning: Log file '{log_file_path}' not found.")
        return "Log file not found."

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
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


def get_global_models(models_file_path):
    models_file_path = Path(models_file_path)
    if not models_file_path.exists():
        print(f"GUI Warning: Models file '{models_file_path}' not found.")
        return "Models file not found."

    try:
        return pd.read_csv(models_file_path)

    except Exception as e:
        print(f"GUI Error: Exception in get_models while loading models from file {models_file_path} :: {e}")
        return None


def get_worker_models(server_config, models_file_path):
    worker_status_list = []

    # Step 1: get worker info
    server_status = get_worker_status(server_config)

    # Step 2: load global models
    global_models_df = get_global_models(models_file_path)
    if global_models_df is None or 'Model' not in global_models_df.columns:
        print("GUI Warning: Global models file missing or malformed.")
        return pd.DataFrame([["Global models config error", "Unable to load models"]], columns=["Worker", "Status"])

    global_models = set(global_models_df['Model'].dropna().astype(str).str.strip())

    # Step 3: For each worker, fetch available models and compare
    for worker_entry in server_status:
        if len(worker_entry) < 2:
            continue
        worker_name = worker_entry[0]
        worker_url = worker_entry[1]

        api_tags_url = f"{worker_url.rstrip('/')}/api/tags"
        try:
            resp = requests.get(api_tags_url, timeout=5)
            if resp.status_code != 200:
                status_msg = f"Fetch Error: HTTP {resp.status_code}"
                worker_status_list.append([worker_name, status_msg, ""])
                continue

            tags_data = resp.json()
            if isinstance(tags_data, dict) and "models" in tags_data:

                worker_models = set(str(m.get("name", "")).strip() for m in tags_data["models"])
            else:
                worker_status_list.append([worker_name, "Fetch Error: Unexpected response format", ""])
                continue

            missing_models = global_models - worker_models
            extra_models = worker_models - global_models

            status = "None" if not missing_models else f"{', '.join(sorted(missing_models))}"
            extra_str = ", ".join(sorted(extra_models)) if extra_models else ""

            worker_status_list.append([worker_name, status, extra_str])

        except Exception as e:
            worker_status_list.append([worker_name, f"Fetch Error: {str(e)}", ""])

    return pd.DataFrame(worker_status_list, columns=["Worker", "Missing", "Additional"])


def get_users(users_file_path):
    _, users = get_authorized_users(users_file_path)
    users = [(user['user'], user['expirationDate'], user['accessKey']) for user in users]
    return pd.DataFrame(users, columns=["user", "expirationDate", "accessKey"])


def add_global_model(models, models_file_path):
    df = pd.DataFrame(models, columns=['Model', 'LastUsed'])
    df.to_csv(models_file_path, index=False, encoding='utf-8')
    return df


def add_user(users, users_file_path):

    users.loc[users["Expiration Date"] == "", "Expiration Date"] = "01.01.0001"
    users.loc[users["Access Key"] == "", "Access Key"] = generate_key()

    users.to_csv(users_file_path, index=False, encoding='utf-8')

    return users


def pull_missing_models(worker_status, server_config):
    logs = []
    server_status = get_worker_status(server_config)

    for worker in worker_status.itertuples(index=False):
        status = getattr(worker, "Missing")
        if status == "None":
            continue

        worker_name = getattr(worker, "Worker")
        url = next((entry[1] for entry in server_status if entry[0] == worker_name), "")

        logs.append([f"[Updating {worker_name}] :: {url}"])

        models = status.split(", ")
        for model in models:
            model_name = model.strip()
            if not model_name:
                continue
            try:
                response = requests.post(
                    url=f"{url}/api/pull",
                    json={"model": model_name},
                    timeout=5
                )

                if response.status_code == 200:
                    try:
                        parts = response.text.strip().split("\n")
                        for part in parts:
                            try:
                                data = json.loads(part)
                                if data.get("status") == "pulling manifest":
                                    continue
                                elif data.get("error"):
                                    logs.append(f"Failed to pull {model_name}: {data['error']}\n")
                                    break
                                elif data.get("status") == "success":
                                    logs.append(f"{model_name} pulled successfully.")
                                    break
                            except json.JSONDecodeError:
                                logs.append(f"Malformed response: {part}")
                    except Exception as e:
                        logs.append(f"Unexpected error while processing {model_name}: {str(e)}")
                else:
                    logs.append(f"Failed to pull {model_name}: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                logs.append(f"Exception while pulling {model_name}: {e}")

    return "\n".join(
        line[0] if isinstance(line, list) else line
        for line in logs
    )


def remove_model(model, server_config):
    logs = []
    for worker in server_config:
        worker_name = worker[0]
        logs.append([worker_name,"NOW PURGING!!!", model])

    return logs


def remove_expired_users(users):
    return 0


def create_gui(server_config, log_file_path, models_file_path, users_file_path):
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
                server_list_output = gr.DataFrame(
                    headers=["Name", "URL", "Queue Size", "Running Models", "GPU Usage", "GPU VRAM Usage",
                             "Ollama Version"],
                    interactive=False,
                    row_count=(10, "dynamic")
                )
                with gr.Row(visible=False) as worker_mangagement:
                    with gr.Column(scale=1):
                        with gr.Row():
                            new_worker_name = gr.Textbox(label="New Worker Name", type="text")
                            new_worker_url = gr.Textbox(label="New Worker URL", type="text")
                        add_worker_btn = gr.Button("Add new Worker")
                    with gr.Column(scale=1):
                        update_ollama_btn = gr.Button("Update Ollama Version")

                gr.Markdown("<br><br>")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Global Models")

                        global_models = gr.DataFrame(
                            headers=["Model", "Last Used"],
                            interactive=False,
                            row_count=(5, "dynamic"),
                            col_count=2

                        )

                        model_name = gr.Textbox(label="Model to add", type="text")
                        add_model_btn = gr.Button("Add new Model")
                        with gr.Row(visible=False) as model_management:
                            remove_model_btn = gr.Button("Remove Model")

                    with gr.Column(scale=1, visible=False) as worker_model_management:
                        gr.Markdown("## Available Models")

                        worker_models = gr.DataFrame(
                            headers=["Worker", "Status", "Additional models"],
                            interactive=False,
                            row_count=(10, "dynamic")
                        )

                        pull_missing_btn = gr.Button("Pull missing models")

                        update_logs = gr.Textbox(
                            label="Log",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True
                        )

            """
                <----- USERS ----->
            """
            with gr.TabItem(label="Users", visible=False) as user_management:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Registered Users")

                        users = gr.DataFrame(
                            headers=["UserID", "Expiration Date"],
                            interactive=False,
                            row_count=(10, "dynamic")
                        )
                        new_user_name = gr.Textbox(label="User", type="text")
                        new_user_expiration = gr.Textbox(label="Optional: Expiration Date (Default: 6 Months)", type="text")
                        add_user_btn = gr.Button("Add new User")
                        gr.Markdown("(Users with past expiration dates will be removed automatically)")

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
                    show_copy_button=True
                )

                with gr.Row():
                    log_lines_input = gr.Number(
                        value=default_log_lines,
                        label="Number of log lines",
                        minimum=1,
                        step=1,
                        precision=0
                    )
                    refresh_logs_btn = gr.Button("Refresh Logs")

                def update_logs_display(lines_to_show_from_input):
                    return get_logs(log_file_path, num_lines=int(lines_to_show_from_input))


        """
            <----- ADMIN LOGIN ----->
        """
        gr.Markdown("---")
        gr.Markdown("#### Admin Login")
        with gr.Row() as admin_login_control:
            password_input = gr.Textbox(label="Password", type="password", placeholder="Enter admin password...")
            login_button = gr.Button("Login")
        login_status = gr.Markdown(visible=False)


        # ------> events <------
        login_button.click(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management, user_management, worker_mangagement, admin_logs, admin_login_control,
                model_management, login_status
            ]
        )
        password_input.submit(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management, user_management, worker_mangagement, admin_logs, admin_login_control,
                model_management, login_status
            ]
        )


        add_user_btn.click(
            fn=lambda users: add_user(users, users_file_path),
            inputs=[users],
            outputs=users
        )

        # TODO
        add_model_btn.click()
        add_worker_btn.click
        update_ollama_btn.click()
        pull_missing_btn.click(
            fn=lambda worker_status: pull_missing_models(worker_status, server_config),
            inputs=[worker_models],
            outputs=update_logs
        )
        remove_model_btn.click(
            fn=lambda model_to_purge: remove_model(model_name, server_config),
            inputs=[model_name],
            outputs=update_logs
        )


        global_models.change(
            fn=lambda: get_worker_models(server_config, models_file_path),
            inputs=None,
            outputs=worker_models
        )
        refresh_logs_btn.click(
            fn=update_logs_display,
            inputs=[log_lines_input],
            outputs=log_output_textbox
        )


        # get server utilization every 3 seconds
        server_status_timer = gr.Timer(value=3.0)
        server_status_timer.tick(
            fn=lambda: get_worker_status(server_config),
            inputs=None,
            outputs=server_list_output
        )

        # read last used every 10 seconds
        last_used_timer = gr.Timer(value=10)
        last_used_timer.tick(
            fn=lambda: get_global_models(models_file_path),
            inputs=None,
            outputs=global_models
        )

        #user_access_timer = gr.Timer(value=300)
        #user_access_timer.tick(
        #    # remove expired users from authorized users
        #)

        # Initial loads
        def on_load():
            server_status = get_worker_status(server_config)
            logs = get_logs(log_file_path, num_lines=default_log_lines)
            models = get_global_models(models_file_path)
            worker_status = get_worker_models(server_config, models_file_path)
            users = get_users(users_file_path)
            return server_status, logs, models, worker_status, users

        demo.load(
            on_load,
            inputs=None,
            outputs=[server_list_output, log_output_textbox, global_models, worker_models, users]
        )

    return demo


def start_gui(gui_port_to_use, server_config, log_file_path, models_file_path, users_file_path):
    """
    Launches the Gradio GUI.
    Passes the server config getter to create_gui.

    launch command: python proxy.py --config ../config.ini --users_list ../authorized_users.txt --log_path access_log.txt --port 8000 --gui_port 7860 --model ../models.txt
    """
    print("GUI: Attempting to launch Gradio GUI...")
    gui_app = create_gui(server_config, log_file_path, models_file_path, users_file_path)
    try:
        gui_app.launch(server_name="localhost", server_port=int(gui_port_to_use), share=False, show_api=False)
        print(f"GUI: Gradio GUI is running on http://localhost:{gui_port_to_use}")
    except Exception as e:
        print(f"GUI Error: Failed to launch Gradio GUI: {e}")


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

    if not args.no_gui:

        start_gui(args.gui_port,
                  SERVERS_CONFIG,
                  LOG_FILE_PATH,
                  MODELS_FILE_PATH,
                  USERS_FILE_PATH)

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

    1. push to correct gitlab repo -- finished
    2. refactor: gui starts server -- finished
    3. test ollama update fastAPI
    4. add ollama config fastAPI
    5. add worker button -> save in config
    6. add_model button api call
    7. make add_modell button public -- finished
    8. add_user button
    9. test using VM (SCS-AI-PROXY)

"""