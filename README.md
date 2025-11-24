# Build and Deploy ML App in Dev with Docker Compose
As an MLOps or AI/ML engineer on the AI Platform Engineering team, the mission is to **simulate a real-world machine learning application** in a development environment by automating its deployment using **Docker Compose**. This setup offers a fast and reliable way to build a **local development environment** and **streamline onboarding** for new data scientists and ML engineers.
<img width="695" height="474" alt="image" src="https://github.com/user-attachments/assets/51f12d24-3506-4822-9050-13acef5bcbd0" />

Create a local dev/test environment that includes:

* **Model training and tracking** with MLflow
* **Model serving** using FastAPI
* **User interaction** through Streamlit

Such an environment enables teams to **test integrations**, **debug workflows**, and **ensure reproducibility** before moving to production systems like **Kubernetes**.

**Workflow steps:**

1. Run `pipeline.py` to execute the pipeline and generate artifacts.
2. Access the **MLflow UI** at [http://localhost:5555](http://localhost:5555).
3. Open the **FastAPI documentation** at [http://localhost:8000/docs](http://localhost:8000/docs).
4. Launch the **Streamlit app** at [http://localhost:8501](http://localhost:8501).
5. Streamlit connects to FastAPI to send requests and display predictions.
6. All services run together using `docker-compose up`.


Using the powerful tool **Docker Compose** for local deployment, testing, and development allows to:

* **Create consistent dev/test environments** across machines
* **Share portable ML applications** with teammates
* **Validate service integrations** before scaling to Kubernetes
* **Work like a real AI/ML Platform Engineering team**

We’re not just containerizing components — we’re simulating a production-grade architecture in a controlled local environment.
