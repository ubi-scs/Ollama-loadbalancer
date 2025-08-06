import argparse
import configparser
import random
import threading
import time
from tempfile import NamedTemporaryFile
from datetime import datetime
from dateutil.relativedelta import relativedelta
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

"""
    ### GUI GETTER/HELPER ###
"""
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

def get_worker_status(worker_config_path):
    status_data = []
    workers = pd.read_csv(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), worker_config_path)))

    try:
        for _, row in workers.iterrows():
            worker_url = row['url']
            worker_name = row['name']
            running_models_str = "N/A"  # Default
            api_ps_url = f"{worker_url.rstrip('/')}/api/ps"

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
                print(f"GUI: Error fetching /api/ps for server {worker_name}: {e_ps}")

            try:
                gpu_util_url = f"{worker_url.replace('11434','8000')}/gpu/utilization"
                response = requests.get(gpu_util_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=5)
                if response.status_code == 200:
                    util_data = response.json()
                    gpu_utilization_str = f"{util_data.get('gpu_utilization_percent', 0.0):.1f}%"
                else:
                    gpu_utilization_str = f"Err {response.status_code}"
            except Exception as e_util:
                gpu_utilization_str = "Fetch Error"
                print(f"GUI: Error fetching GPU utilization for server {worker_name}: {e_util}")

                # Fetch VRAM usage
            try:
                vram_url = f"{worker_url.replace('11434','8000')}/gpu/vram"
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
                print(f"GUI: Error fetching VRAM info for server {worker_name}: {e_vram}")

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
                print(f"GUI: Error fetching Ollama version info for server {row['name']}: {e_version}")

            status_data.append([worker_name, worker_url,
                                running_models_str,
                                gpu_utilization_str,
                                vram_usage_str,
                                ollama_version])


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
    log_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file_path_str))

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
    models_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), models_file_path))
    return pd.read_csv(models_file_path)

def get_worker_models(worker_config_path, models_file_path):
    worker_status_list = []

    # Step 1: get worker info
    server_status = get_worker_status(worker_config_path)

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
    users_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), users_file_path))
    df = pd.read_csv(users_file_path)
    return pd.DataFrame(df, columns=["user", "expirationDate", "accessKey"])


"""
   ### GUI INTERACTION ###
"""
def add_global_model(models_file_path, name):
    models_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), models_file_path))

    new_model = pd.DataFrame([{'Model': name, 'LastUsed': datetime.today().strftime("%d.%m.%Y")}])
    models = pd.concat([pd.read_csv(models_file_path), new_model], ignore_index=True)

    models.to_csv(models_file_path, index=False, encoding='utf-8')
    return models

def add_user(users_file_path, new_user_name, new_user_expiration=None):
    users_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), users_file_path))

    def generate_key(length=10):
        """Generate a random key of given length"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;,.<>?/~'
        return ''.join(random.choice(chars) for _ in range(length))

    if not new_user_expiration:
        new_user_expiration = (datetime.today() + relativedelta(months=6)).strftime("%d.%m.%Y")

    new_user = pd.DataFrame([{'user': new_user_name, 'expirationDate': new_user_expiration, 'accessKey': generate_key()}])
    users = pd.concat([pd.read_csv(users_file_path), new_user], ignore_index=True)

    users.to_csv(users_file_path, index=False, encoding='utf-8')
    return users

def add_worker(workers_file_path, new_worker_name, new_worker_url):
    workers = pd.read_csv(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), workers_file_path)))

    new_worker = pd.DataFrame([{'name': new_worker_name, 'url': new_worker_url, 'enabled': True}])
    workers = pd.concat([workers, new_worker], ignore_index=True)
    workers.to_csv(workers_file_path, index=False, encoding='utf-8')

    return get_worker_status(workers_file_path)

def pull_missing_models(worker_status, worker_config_path):
    logs = []
    server_status = get_worker_status(worker_config_path)

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

def remove_model(model, worker_config_path):
    logs = []
    workers = pd.read_csv(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), worker_config_path)))
    for _, row in workers:
        worker_name = row["name"]
        logs.append([worker_name,"NOW PURGING!!!", model])

    return logs

def update_ollama(worker_config_path):
    return 0

def create_gui(worker_config_path, log_file_path, models_file_path, authorized_users_path):
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
                server_list = gr.DataFrame(
                    headers=["Name", "URL", "Running Models", "GPU Usage", "GPU VRAM Usage",
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
            fn=lambda name, expiration: add_user(authorized_users_path, name, new_user_expiration=expiration),
            inputs=[new_user_name, new_user_expiration],
            outputs=users
        )

        add_model_btn.click(
            fn=lambda model_name: add_global_model(models_file_path, model_name),
            inputs=[model_name],
            outputs=global_models
        )

        add_worker_btn.click(
            fn=lambda new_worker_name, new_worker_url: add_worker(worker_config_path, new_worker_name, new_worker_url),
            inputs=[new_worker_name, new_worker_url],
            outputs=server_list
        )

        update_ollama_btn.click(
            fn=lambda: update_ollama(worker_config_path),
            inputs=None,
            outputs=server_list
        )

        pull_missing_btn.click(
            fn=lambda worker_status: pull_missing_models(worker_status, worker_config_path),
            inputs=[worker_models],
            outputs=update_logs
        )

        remove_model_btn.click(
            fn=lambda model_to_purge: remove_model(model_name, worker_config_path),
            inputs=[model_name],
            outputs=update_logs
        )

        global_models.change(
            fn=lambda: get_worker_models(worker_config_path, models_file_path),
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
            fn=lambda: get_worker_status(worker_config_path),
            inputs=None,
            outputs=server_list
        )

        # read last used every 10 seconds
        last_used_timer = gr.Timer(value=10)
        last_used_timer.tick(
            fn=lambda: get_global_models(models_file_path),
            inputs=None,
            outputs=global_models
        )



        # Initial loads
        def on_load():
            server_status = get_worker_status(worker_config_path)
            logs = get_logs(log_file_path, num_lines=default_log_lines)
            models = get_global_models(models_file_path)
            worker_status = get_worker_models(worker_config_path, models_file_path)
            users = get_users(authorized_users_path)
            return server_status, logs, models, worker_status, users

        demo.load(
            on_load,
            inputs=None,
            outputs=[server_list, log_output_textbox, global_models, worker_models, users]
        )

    return demo

"""
    ### GENERAL ###
"""
def clean_expired_users():
    file_path = 'authorized_users.csv'
    today = datetime.today().date()
    temp_file = NamedTemporaryFile(mode='w', delete=False, newline='')

    with open(file_path, mode='r', newline='') as csvfile, temp_file:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                exp_date = datetime.strptime(row['expirationDate'], '%d.%m.%Y').date()
                if exp_date >= today:
                    writer.writerow(row)
            except ValueError as e:
                print(f"Skipping row due to error: {e}")

    os.replace(temp_file.name, file_path)

def start_cleanup_thread(intervall_hours=2):
    sleep_time_seconds = intervall_hours * 60 * 60
    def loop():
        while True:
            clean_expired_users()
            time.sleep(sleep_time_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread

def start_gui(gui_port_to_use, server_config, log_file_path, models_file_path, users_file_path):
    """
    Launches the Gradio GUI.
    Passes the server config getter to create_gui.

    launch command: python proxy.py --config ../workers.csv --users_list ../authorized_users.csv --log_path access_log.txt --port 8000 --gui_port 7860 --model ../models.txt
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
    parser.add_argument('--config', default="workers.csv", help='Path to the server configuration file (default: workers.csv)')
    parser.add_argument('--log_path', default="access_log.txt", help='Path to the access log file (default: access_log.txt)')
    parser.add_argument('--users_list', default="authorized_users.csv", help='Path to the authorized users list file (default: authorized_users.csv)')
    parser.add_argument('--models', default="models.txt", help='Models available on all workers (default: models.txt)')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the proxy server (default: 8000)')
    parser.add_argument('--gui_port', type=int, default=7860, help='Port number for the Gradio GUI (default: 7860)')
    parser.add_argument('-d', '--deactivate_security', action='store_true', help='Deactivates security layer (USE WITH CAUTION)')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.config
    USERS_FILE_PATH = args.users_list
    LOG_FILE_PATH = args.log_path
    DEACTIVATE_SECURITY = args.deactivate_security
    MODELS_FILE_PATH = args.models

    print("Ollama Proxy server")
    print(f"Configuration file: {CONFIG_FILE_PATH}")
    print(f"Users list file: {USERS_FILE_PATH}")
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"Models file: {MODELS_FILE_PATH}")

    proxy_thread = threading.Thread(target=start_proxy_server, args=(args.port, RequestHandler), daemon=True)
    proxy_thread.start()

    start_cleanup_thread(intervall_hours=2)

    start_gui(args.gui_port,
              CONFIG_FILE_PATH,
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

    1. test ollama update fastAPI
    2. add ollama config fastAPI
    
    6. test using VM (SCS-AI-PROXY)
    7. refactor all conifguration files
"""