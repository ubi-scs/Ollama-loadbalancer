## Ollama Proxy Server

```bash
cd /opt/
sudo git clone https://git@gitlab.ub.uni-bielefeld.de/scs/enrico/ollama-load-balancer-interface.git
cd ollama-load-balancer-interface/worker_application
sudo chmod +x setup_environment.sh
sudo apt install python3.8-venv
sudo ./setup_environment.sh
cp ollama_helper.service /etc/systemd/system/
```

See you space cowboy!

