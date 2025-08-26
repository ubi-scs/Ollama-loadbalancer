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

from proxy import RequestHandler, ThreadedHTTPServer, LOG_FILE_PATH


CORRECT_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not CORRECT_PASSWORD:
    raise RuntimeError(f"ADMIN_PASSWORD environment variable not set. Please provide a secret Admin password.")

OLLAMA_HELPER_API_KEY = os.environ.get("OLLAMA_HELPER_API_KEY")
if not OLLAMA_HELPER_API_KEY:
    raise RuntimeError(f"OLLAMA_HELPER_API_KEY environment variable not set. Please provide a secret OLLAMA_HELPER_API_KEY.")


WORKER_CONFIG_PATH = 'workers.csv'
AUTHORIZED_USERS_CONFIG_PATH = 'authorized_users.csv'
MODELS_CONFIG_PATH = 'models.csv'


def get_worker_status():
    status_data = []


    workers = pd.read_csv(WORKER_CONFIG_PATH)

    try:
        for _, row in workers.iterrows():
            worker_url = row['url']
            worker_name = row['name']
            if not row['enabled']:
                status_data.append([worker_name, worker_url, 'disabled', 'disabled', 'disabled', 'disabled',
                                    row['enabled']])
                continue

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
                gpu_util_url = f"{worker_url.replace('11434','18034')}/gpu/utilization"
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
                vram_url = f"{worker_url.replace('11434','18034')}/gpu/vram"
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
                                ollama_version,
                                row['enabled']])


        if not status_data:
            print("GUI: current_servers_config was present, but status_data is empty. Returning default empty row.")
            return [["No valid server data found.", "", "", ""]]
        return status_data
    except Exception as e:
        print(f"GUI Error: Exception in get_server_status_for_gui: {e}")
        import traceback
        traceback.print_exc()
        return [["Error processing server data.", str(e), "", ""]]


def get_logs(num_lines):
    """
    Reads the last N lines from the access log file.
    """
    log_file_path = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_PATH))

    try:
        if not os.path.exists(log_file_path):
            return "Log file not found."
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


def get_global_models():
    #should look like this: Model,LastUsed
    if not os.path.exists(MODELS_CONFIG_PATH):
        print("GUI Warning: Models config file not found. Creating new file.")
        pd.DataFrame(columns=['Model', 'LastUsed']).to_csv(MODELS_CONFIG_PATH, index=False, encoding='utf-8')
    return pd.read_csv(MODELS_CONFIG_PATH)


def get_worker_models():
    worker_status_list = []

    # Step 1: get worker info
    server_status = get_worker_status()

    # Step 2: load global models
    global_models_df = get_global_models()
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


def get_users():
    #create if not exists. Structure: user,expirationDate,accessKey
    if not os.path.exists(AUTHORIZED_USERS_CONFIG_PATH):
        print("GUI Warning: Authorized users config file not found. Creating new file.")
        pd.DataFrame(columns=['user', 'expirationDate', 'accessKey']).to_csv(AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding='utf-8')
    return pd.read_csv(AUTHORIZED_USERS_CONFIG_PATH)


def add_global_model(model_name):

    worker_status = get_worker_status()
    for worker in worker_status:

        worker_name = worker[0]
        url = worker[1]

        if not url:
            continue

        gr.Info(f"Trying to add model {model_name} to {worker_name}")

        #if model name contains :latest, show warning and skip
        if ":latest" in model_name:
            gr.Warning(f"  - Skipping {model_name} on {worker_name}: 'latest' tag not allowed. Please specify actual size so that we can manage model sizes properly.")
            continue
        #same if size is not specified ie does not contain : with a number and b at the end
        if not re.search(r":\d+[mb]$", model_name):
            gr.Warning(f"  - Skipping {model_name} on {worker_name}: Please specify model size in billions, e.g. modelname:4b")
            continue

        last_reported_percent = -1
        try:
            response = requests.post(
                url=f"{url}/api/pull",
                json={"model": model_name, "stream": True},
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode('utf-8'))

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
                                    percent_done >= 99 and last_reported_percent < 99):
                                last_reported_percent = percent_done
                                gr.Info(f"  - Downloading... {percent_done:.1f}%")

                    if data.get("status") == "success":

                        gr.Info(f"  + {model_name} pulled successfully to {worker_name}.")


                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        except requests.exceptions.RequestException as e:
            gr.Info(f"  - EXCEPTION for {model_name}: {e}")

    gr.Info("Update process finished.")

    new_model = pd.DataFrame([{'Model': model_name, 'LastUsed': datetime.today().strftime("%d.%m.%Y")}])
    models = pd.concat([pd.read_csv(MODELS_CONFIG_PATH), new_model], ignore_index=True)

    models.to_csv(MODELS_CONFIG_PATH, index=False, encoding='utf-8')
    return models


def add_user(new_user_name, new_user_expiration=None):

    def generate_key(length=20):
        """Generate a random key of given length"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        return ''.join(random.choice(chars) for _ in range(length))

    if not new_user_expiration:
        new_user_expiration = (datetime.today() + relativedelta(months=6)).strftime("%d.%m.%Y")

    new_user = pd.DataFrame([{'user': new_user_name, 'expirationDate': new_user_expiration, 'accessKey': generate_key()}])
    users = pd.concat([pd.read_csv(AUTHORIZED_USERS_CONFIG_PATH), new_user], ignore_index=True)

    users.to_csv(AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding='utf-8')
    return users


def add_worker(new_worker_name, new_worker_url):

    if new_worker_url == '':
        return get_worker_status()

    workers = pd.read_csv(WORKER_CONFIG_PATH)

    new_worker = pd.DataFrame([{'name': new_worker_name, 'url': new_worker_url, 'enabled': True}])
    workers = pd.concat([workers, new_worker], ignore_index=True)
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding='utf-8')

    return get_worker_status()


def add_missing_models_generator():
    logs = []
    server_status = get_worker_status()
    worker_models = get_worker_models()

    for worker in worker_models.itertuples(index=False):
        status = getattr(worker, "Missing")
        if not status or status == "None":
            continue

        worker_name = getattr(worker, "Worker")
        url = next((entry[1] for entry in server_status if entry[0] == worker_name), "")

        if not url:
            logs.append(f"[Skipping {worker_name}] :: URL not found")
            yield "\n".join(logs)
            continue

        logs.append(f"[Updating {worker_name}]")
        yield "\n".join(logs)

        models = [model.strip() for model in status.split(',') if model.strip()]

        for model_name in models:
            last_reported_percent = -1
            try:
                response = requests.post(
                    url=f"{url}/api/pull",
                    json={"model": model_name, "stream": True},
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                        log_line = ""
                        if data.get("error"):
                            log_line = f"  - Failed to pull {model_name}: {data['error']}"
                        elif "pulling manifest" in data.get("status", ""):
                            log_line = f"  - Pulling {model_name}:"
                        elif "total" in data and "completed" in data.keys():
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            if total > 0:
                                percent_done = (completed / total) * 100
                                if percent_done - last_reported_percent >= 10 or (
                                        percent_done >= 99 and last_reported_percent < 99):
                                    last_reported_percent = percent_done
                                    log_line = f"  - Downloading... {percent_done:.1f}%"
                        if log_line and logs[-1] != log_line:
                            logs.append(log_line)
                            yield "\n".join(logs)

                        if data.get("status") == "success":
                            logs.append(f"  + {model_name} pulled successfully to {worker_name}.")
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
    users = users[users['user'] != user_name]

    users.to_csv(AUTHORIZED_USERS_CONFIG_PATH, index=False, encoding='utf-8')
    return users


def remove_model(model_name):
    models = pd.read_csv(MODELS_CONFIG_PATH)

    models = models[models['Model'] != model_name]
    models.to_csv(MODELS_CONFIG_PATH, index=False, encoding='utf-8')

    logs = []
    server_status = get_worker_status()


    logs.append(f"Removing: {model_name}")
    yield "\n".join(logs)

    for worker_name, url, *_ in server_status:

        try:
            response = requests.delete(
                url=f"{url}/api/delete",
                json={"name": model_name},
                timeout=120
            )

            if response.status_code == 404:
                logs.append(f"  - Model '{model_name}' not found on {worker_name} (already removed).")
                yield "\n".join(logs)
                continue

            response.raise_for_status()
            logs.append(f"  + Successfully removed '{model_name}' from {worker_name}.")
            yield "\n".join(logs)

        except requests.exceptions.RequestException as e:
            logs.append(f"  - EXCEPTION for {worker_name}: {e}")
            yield "\n".join(logs)

    logs.append("\nRemoval process finished.")
    yield "\n".join(logs)


def remove_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers = workers[workers['name'] != worker_name]
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding='utf-8')

    return get_worker_status()


def disable_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers.loc[workers['name'] == worker_name, 'enabled'] = False
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding='utf-8')

    return get_worker_status()


def enable_worker(worker_name):
    workers = pd.read_csv(WORKER_CONFIG_PATH)

    workers.loc[workers['name'] == worker_name, 'enabled'] = True
    workers.to_csv(WORKER_CONFIG_PATH, index=False, encoding='utf-8')

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
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="Login successful! Admin controls enabled.", visible=True)
        )
    else:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(value="Incorrect password.", visible=True)
        )


def update_ollama(worker_status):
    """
    Calls the /ollama/update endpoint on each enabled worker and displays results.
    Uses port 18034 for the worker application.
    """
    logs = []
    #worker_status is a dataframe
    for worker in worker_status.itertuples(index=False):
        worker_name = worker[0]
        worker_url = worker[1]
        enabled = worker[-1]
        if not enabled or str(enabled).lower() == "false":
            logs.append(f"[{worker_name}] Skipped (disabled)")
            continue

        # Validate URL scheme
        if not isinstance(worker_url, str) or not re.match(r'^https?://', worker_url):
            logs.append(f"[{worker_name}] Skipped: Invalid URL '{worker_url}' (no scheme supplied)")
            continue

        try:
            # Ensure the worker_url uses port 18034
            if "://" in worker_url:
                proto, rest = worker_url.split("://", 1)
                if ":" in rest:
                    host, *path = rest.split("/", 1)
                    host = re.sub(r":\d+", ":18034", host)
                    new_url = f"{proto}://{host}"
                    if path:
                        new_url += "/" + path[0]
                else:
                    new_url = f"{proto}://{rest}:18034"
            else:
                new_url = worker_url

            update_url = f"{new_url.rstrip('/')}/ollama/update"
            response = requests.post(update_url, headers={"x-api-key": OLLAMA_HELPER_API_KEY}, timeout=10)
            if response.status_code == 202:
                msg = response.json().get("message", "Update started.")
                logs.append(f"[{worker_name}] Update triggered: {msg}")
            else:
                logs.append(f"[{worker_name}] Error: HTTP {response.status_code} - {response.text}")
        except Exception as e:
            logs.append(f"[{worker_name}] Exception: {e}")
    gr.Info("\n".join(logs))
    # Optionally, refresh worker status after update
    return get_worker_status()


def clean_expired_users():
    today = datetime.today().date()
    temp_file = NamedTemporaryFile(mode='w', delete=False, newline='')
    get_users()  # Ensure the file exists

    with open(AUTHORIZED_USERS_CONFIG_PATH, mode='r', newline='') as csvfile, temp_file:
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

    os.replace(temp_file.name, AUTHORIZED_USERS_CONFIG_PATH)


def start_cleanup_thread(intervall_hours=2):
    sleep_time_seconds = intervall_hours * 60 * 60
    def loop():
        while True:
            clean_expired_users()
            time.sleep(sleep_time_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


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
                    headers=["Name", "URL", "Running Models", "GPU Usage", "GPU VRAM Usage",
                             "Ollama Version", "Enabled"],
                    interactive=False,
                    row_count=(10, "dynamic")
                )
                with gr.Row(visible=False) as worker_mangagement:
                    with gr.Column(scale=1):
                        with gr.Row():
                            new_worker_name = gr.Textbox(label="Worker Name", type="text")
                            new_worker_url = gr.Textbox(label="Worker URL", type="text")
                        with gr.Row():
                            with gr.Column(scale=1):
                                add_worker_btn = gr.Button("Add Worker")
                                remove_worker_btn = gr.Button("Remove Worker")
                            with gr.Column(scale=1):
                                enable_worker_btn = gr.Button("Enable")
                                disable_worker_btn = gr.Button("Disable")
                        update_ollama_btn = gr.Button("Update Ollama Version")
                    with gr.Column(scale=1):
                        ollama_update_log = gr.Textbox(
                            label="Ollama Update Log",
                            lines=10,
                            max_lines=10,
                            interactive=False,
                            show_copy_button=True
                        )

                gr.Markdown("<br>")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Global Models")

                        global_models = gr.DataFrame(
                            headers=["Model", "Last Used"],
                            interactive=False,
                            row_count=(5, "dynamic"),
                            col_count=2

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

                gr.Markdown("## Registered Users")

                users = gr.DataFrame(
                    headers=["UserID", "Expiration Date"],
                    interactive=True,
                    row_count=(10, "dynamic")
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        new_user_name = gr.Textbox(label="User", type="text")
                        new_user_expiration = gr.Textbox(label="Optional: Expiration Date (Default: 6 Months)", type="text")
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
                    return get_logs(num_lines=int(lines_to_show_from_input))


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
        password_input.submit(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management, user_management, worker_mangagement, admin_logs, admin_login_control,
                model_management, login_status
            ]
        )

        login_button.click(
            fn=login,
            inputs=password_input,
            outputs=[
                worker_model_management, user_management, worker_mangagement, admin_logs, admin_login_control,
                model_management, login_status
            ]
        )

        add_user_btn.click(
            fn=lambda name, expiration: add_user(name, new_user_expiration=expiration),
            inputs=[new_user_name, new_user_expiration],
            outputs=users
        )

        remove_user_btn.click(
            fn=lambda name: remove_user(name),
            inputs=[new_user_name],
            outputs=users
        )

        add_worker_btn.click(
            fn=lambda new_worker_name, new_worker_url: add_worker(new_worker_name, new_worker_url),
            inputs=[new_worker_name, new_worker_url],
            outputs=worker_status
        )

        add_model_btn.click(
            fn=lambda model_name: add_global_model(model_name),
            inputs=[model_name],
            outputs=global_models
        )

        disable_worker_btn.click(
            fn= lambda edit_worker_name: disable_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status
        )

        enable_worker_btn.click(
            fn= lambda edit_worker_name: enable_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status
        )

        remove_worker_btn.click(
            fn=lambda edit_worker_name: remove_worker(edit_worker_name),
            inputs=[new_worker_name],
            outputs=worker_status
        )

        update_ollama_btn.click(
            fn=update_ollama,
            inputs=[worker_status],
            outputs=worker_status
        )

        pull_missing_btn.click(
            fn=add_missing_models_generator,
            inputs=None,
            outputs=update_logs
        )

        remove_model_btn.click(
            fn=remove_model,
            inputs=[model_name],
            outputs=update_logs
        )

        refresh_logs_btn.click(
            fn=update_logs_display,
            inputs=[log_lines_input],
            outputs=log_output_textbox
        )

        global_models.change(
            fn=get_worker_models,
            inputs=None,
            outputs=worker_models
        )

        status_timer = gr.Timer(value=3.0)
        status_timer.tick(
            fn=get_worker_status,
            inputs=None,
            outputs=worker_status
        )
        status_timer.tick(
            fn=get_worker_models,
            inputs=None,
            outputs=worker_models
        )
        status_timer.tick(
            fn=get_global_models,
            inputs=None,
            outputs=global_models
        )

        # Initial loads
        def on_load():
            server_status = get_worker_status()
            logs = get_logs(num_lines=default_log_lines)
            models = get_global_models()
            worker_status = get_worker_models()
            users = get_users()
            return server_status, logs, models, worker_status, users

        demo.load(
            on_load,
            inputs=None,
            outputs=[worker_status, log_output_textbox, global_models, worker_models, users]
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
        gui_app.launch(server_name="0.0.0.0", server_port=int(gui_port_to_use), share=False, show_api=False)
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
    global WORKER_CONFIG_PATH, AUTHORIZED_USERS_CONFIG_PATH, MODELS_PATH

    parser = argparse.ArgumentParser(description="Ollama Proxy Server with Security and Load Balancing")
    parser.add_argument('--config', default="workers.csv", help='Path to the server configuration file (default: workers.csv)')
    parser.add_argument('--log_path', default="access_log.txt", help='Path to the access log file (default: access_log.txt)')
    parser.add_argument('--users_list', default="authorized_users.csv", help='Path to the authorized users list file (default: authorized_users.csv)')
    parser.add_argument('--models', default="models.csv", help='Models available on all workers (default: models.csv)')
    parser.add_argument('--port', type=int, default=11434, help='Port number for the proxy server (default: 8000)')
    parser.add_argument('--gui_port', type=int, default=7860, help='Port number for the Gradio GUI (default: 7860)')
    args = parser.parse_args()

    WORKER_CONFIG_PATH = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config))
    AUTHORIZED_USERS_CONFIG_PATH = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.users_list))
    LOG_FILE_PATH = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_path))
    MODELS_PATH = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.models))

    if not os.path.exists(WORKER_CONFIG_PATH):
        print("GUI Warning: Workers config file not found. Creating new file.")
        pd.DataFrame(columns=['name', 'url', 'enabled']).to_csv(WORKER_CONFIG_PATH, index=False, encoding='utf-8')
    get_users()
    get_global_models()
    get_logs(1)

    print("Ollama Proxy server")
    print(f"Configuration file: {WORKER_CONFIG_PATH}")
    print(f"Users list file: {AUTHORIZED_USERS_CONFIG_PATH}")
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"Models file: {MODELS_PATH}")

    proxy_thread = threading.Thread(target=start_proxy_server, args=(args.port, RequestHandler), daemon=True)
    proxy_thread.start()

    start_cleanup_thread(intervall_hours=2)

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