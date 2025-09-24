## Ollama Proxy Worker

Install the ollama helper worker in each node where ollama is installed.
```bash
cd /opt/
sudo git clone https://git@gitlab.ub.uni-bielefeld.de/scs/enrico/ollama-load-balancer-interface.git
cd ollama-load-balancer-interface/worker_application
sudo chmod +x setup_environment.sh
sudo apt install python3.*-venv
sudo ./setup_environment.sh
sudo cp ollama_helper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ollama_helper.service
sudo systemctl start ollama_helper.service
```

Afterwards modify the ollama_worker.env to contain the correct port and the OLLAMA_HELPER_API_KEY that is needed for the server to authenticate correctly.
If the worker was not installed before and ollama is not open to the network, please add the worker to the server console and press "Update Ollama version" once to set all environment variables correctly.

See you space cowboy!

