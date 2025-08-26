FROM python:3.11

# Update packagtes, install necessary tools into the base image, clean up and clone git repository
RUN apt update \
    && apt install -y --no-install-recommends --no-install-suggests git \
    && apt autoremove -y --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \

# Change working directory to cloned git repository
WORKDIR ollama_proxy_server

COPY requirements.txt .
COPY requirements_dev.txt .

# Install all needed requirements
RUN pip3 install -r requirements.txt

# Copy everything from local directory to working directory in the container
COPY .. .

# Do not buffer output, e.g. logs to stdout
ENV PYTHONUNBUFFERED=1
ENV ADMIN_PASSWORD="asdf1234"
ENV OLLAMA_HELPER_API_KEY="1234asdf"

RUN python3 gui.py
