# BusGuard AI — Complete Deployment Guide

This guide details four production-ready deployment options for **BusGuard AI**, ranging from zero-cost public cloud hosting to Docker containerization and instant live demonstration tunnels.

---

## Deployment Options Overview

| Option | Platform | Setup Time | Cost | Best For |
|:---|:---|:---:|:---:|:---|
| **Option 1 (Recommended)** | **Streamlit Community Cloud** | 2 Mins | Free | Competition evaluation, live judge links, portfolio |
| **Option 2** | **Docker / Docker Compose** | 3 Mins | Free | On-premise industrial servers, private cloud VMs |
| **Option 3** | **Render / Railway / Cloud Run** | 5 Mins | Free / Low | Production cloud PaaS with custom domains |
| **Option 4** | **Instant Web Tunnel (ngrok / localtunnel)** | 30 Secs | Free | Immediate live judge testing during presentations |

---

## Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud connects directly to your GitHub repository and provides free continuous hosting with an instant public `https://*.streamlit.app` URL.

### Step 1: Push Project to GitHub
Initialize a git repository (if not already done) and push to GitHub:
```bash
git init
git add .
git commit -m "feat: complete BusGuard AI production MVP"
git branch -M main
git remote add origin https://github.com/<your-username>/BusGuard_AI.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in the deployment fields:
   - **Repository**: `<your-username>/BusGuard_AI`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Click **"Deploy!"**.
5. Within 90 seconds, your app will be live at a public URL like `https://busguard-ai.streamlit.app`.

---

## Option 2: Docker / Docker Compose Deployment

The codebase includes an enterprise-ready [`Dockerfile`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/Dockerfile) and [`docker-compose.yml`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/docker-compose.yml) configured with non-root user permissions, layer caching, and healthcheck probes.

### Single-Command Startup:
```bash
docker compose up -d --build
```

### Verify Running Container:
```bash
docker ps
docker logs -f busguard-ai-monitor
```
Access the application at: **`http://localhost:8501`** (or `http://<server-ip>:8501`).

### To Stop:
```bash
docker compose down
```

---

## Option 3: Render or Railway Cloud PaaS

### Render (Free Web Service)
1. Sign in to [render.com](https://render.com).
2. Click **"New +"** -> **"Web Service"** and select your GitHub repository.
3. Configure the service:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **"Create Web Service"**.

### Railway
1. Sign in to [railway.app](https://railway.app).
2. Click **"New Project"** -> **"Deploy from GitHub repo"**.
3. Railway automatically detects the `Dockerfile` or `requirements.txt` and provisions a public HTTPS domain.

---

## Option 4: Instant Public Tunnel for Live Presentations (30 Seconds)

If you are presenting to judges right now and want them to access the dashboard on their phones or laptops without deploying to the cloud first, expose your local instance securely:

### Using localtunnel (No sign-up required):
```bash
npx localtunnel --port 8501
```
*Gives you an instant public URL (e.g. `https://busguard-ai-demo.loca.lt`) accessible from any device worldwide.*

### Using ngrok:
```bash
ngrok http 8501
```
*Provides a secure public HTTPS forwarding URL.*

---

## Production Security & Portability Checklist

- [x] **Zero Hardcoded Paths**: Dynamic relative paths (`os.path.abspath`) ensure compatibility across Windows, Linux, and Docker containers.
- [x] **Safe Memory Footprint**: Models and features run smoothly inside containers with as little as 512 MB RAM.
- [x] **Streamlit Server Config**: Headless mode, dark theme, and CORS policies configured in [`.streamlit/config.toml`](file:///c:/Users/Lenovo/OneDrive/BusGuard_AI/.streamlit/config.toml).
- [x] **Docker Layer Caching**: Pre-trained model artifacts and requirements are cached for near-instant container builds.

