# Resolvei Retrieval Server

Uvicorn server with backend and Retrieval-Augmented Generation logic. Other services are running using Docker Compose. This repository is one fork of [DMP-AIHelpdeskSystem](https://github.com/marekruttner/DMP-AIHelpdeskSystem), other parts of the system can be found in other repositories on my profile. 

## Minimal Requirements 
- **GPU(with CUDA support):** nvidia GeForce RTX 4070 (12 GB VRAM)
- **Memory:** 16 GB 
- **Disk Space:** minimal 20 GB 
- Ubuntu 24.0.2 LTS (tested on this version) or Windows 11 (tested on this version)

## Startup

### Docker compose + separate uvicorn server 
#### Requirements:
- Python 3.12
- Installed [Ollama](https://ollama.com/download/linux)

#### First installation:
- Create python virtual environment and install dependencies 
  - `sudo apt isntall python3-venv` to isntall virtual env
  -  `python3 -m venv venv` to create virtual env
  - `source venv/bin/activate` to activate virtual env
  - `pip install -r requirements.txt` to install all dependencies 
- Start all containers with `docker compose up -d`
- `uvicorn frontend:app --host 0.0.0.0 --port 8000` to start uvicorn server
- Build frontend application from [resolvei-application ](https://github.com/marekruttner/resolvei-application) or try demo here

#### Other startups 
- `source venv/bin/activate` to activate virtual env
- Start all containers with `docker compose up -d`
- `uvicorn frontend:app --host 0.0.0.0 --port 8000` to start uvicorn server
- Build frontend application from [resolvei-application ](https://github.com/marekruttner/resolvei-application) or try demo here

### Docker compose 
- Start all containers with `docker compose up -d`

### Kubernetes 

### Kubernetes (CPU only)


---
Developed by [Marek Ruttner](https://cz.linkedin.com/in/marek-ruttner) 2025