# End-to-End ML Application Stack with MLflow, FastAPI and Streamlit

This project demonstrates a **complete machine learning application stack** running in a local development environment using **Docker Compose** and **Kubernetes**.

The objective is to simulate a **real-world ML system** where model training, experiment tracking, model serving, and user interaction are integrated into a **single reproducible workflow**.

<img src="https://raw.githubusercontent.com/mehdirezvandehy/end-to-end-ml-app-stack-with-mlflow-fastapi-streamlit/main/assets/deployment_schematic.gif" width="600" height="400">

<img src="https://raw.githubusercontent.com/mehdirezvandehy/end-to-end-ml-app-stack-with-mlflow-fastapi-streamlit/main/assets/fig1-3.png"
     width="500" height="500"
     style="display: block; margin-left: 1000px;">

This project creates a **local development and testing environment** that includes:

* **Model training and experiment tracking** with MLflow
* **Model serving** using FastAPI
* **User interaction and visualization** through Streamlit

Such an environment allows teams to **test service integrations**, **debug workflows**, and **ensure reproducibility** before deploying to production platforms such as **Kubernetes**.

Using **Docker Compose** for local development and testing enables teams to:

* **Create consistent development environments** across machines
* **Share portable ML applications** with collaborators
* **Validate service integrations** before scaling to orchestration platforms like Kubernetes
* **Develop workflows similar to those used in real AI/ML platform engineering teams**

Docker Compose orchestrates these services and runs them together as a unified application.

Finally, **simulate production-like orchestration locally using Kubernetes**:

* Run Kubernetes locally with Minikube/Kind/k3d
* Build Docker images before deploying to cluster
* Load images into Minikube or push to registry
* Use kubectl to deploy and manage resources
* Verify pods/services and access apps via Minikube

---

# Prerequisites

Before running the project, install **Docker Desktop**.

Download it from:

[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

Docker Desktop includes:

* Docker Engine
* Docker CLI
* Docker Compose

After installation, verify Docker is working:

```bash
docker --version
docker compose version
```
---

# Clone the Repository

```bash
git clone https://github.com/MehdiRezvandehy/end-to-end-ml-app-stack-with-docker-compose.git

cd end-to-end-ml-app-stack-with-docker-compose
```

---
# Workflow

### 1. Run the Machine Learning Pipeline

Execute the pipeline to train the model and log artifacts to MLflow.

```bash
python pipeline.py
```

This step:

* trains the model
* logs parameters and metrics
* stores artifacts in MLflow

---

### 2. Access MLflow

View experiments, metrics, and artifacts in the MLflow interface:

[http://localhost:5555](http://localhost:5555)

---

### 3. Test the FastAPI Service

Open the FastAPI interactive API documentation:

[http://localhost:8000/docs](http://localhost:8000/docs)

---

### 4. Launch the Streamlit Application

Interact with the model through the Streamlit interface:

[http://localhost:8501](http://localhost:8501)

---

### Service Interaction

The system components communicate as follows:

* **Streamlit** sends prediction requests to **FastAPI**
* **FastAPI** loads the trained model from **MLflow artifacts**
* Predictions are returned to **Streamlit** and displayed to the user

---

## Project Structure

```
project-root
│ 
├── pipeline.py          # ML training pipeline
├── fastapi/             # FastAPI model serving
├── streamlit/           # Streamlit frontend
├── docker-compose.yml   # Multi-container orchestration
├── Dockerfile           # Container build instructions
├── k8s/                 # Kubernetes manifests
└── README.md
```

---

## ☸️ Run with Kubernetes (Minikube)

This project also supports **Kubernetes deployment**, simulating production environments.

### 1. Start Minikube

```bash
minikube start
```

### 2. Apply Kubernetes manifests

```bash
kubectl apply -f k8s/
```

### 3. Check resources

```bash
kubectl get pods
kubectl get services
```

### 4. Access services

```bash
minikube service <service-name>
```

---

### ✅ Why Kubernetes?

Using Kubernetes allows you to:

* **Deploy ML systems in a production-like environment**
* **Enable scaling and load balancing**
* **Improve fault tolerance and reliability**
* **Standardize deployment across cloud platforms**

---


## 🎯 Key Takeaways

* ML systems are more than models — they require **infrastructure and orchestration**
* APIs (FastAPI) make models **production-ready**
* MLflow ensures **reproducibility and versioning**
* Kubernetes brings **scalability and reliability**


