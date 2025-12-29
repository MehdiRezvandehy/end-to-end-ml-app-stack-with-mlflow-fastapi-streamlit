import subprocess
import sys

# Use the current environment's Python executable
python_executable = sys.executable
print (f"Current environment's Python executable: {python_executable}")

print('# ----------------------------------------------------------------------------------------------')
print('#                                1. Train the model                                             ')
print('# ----------------------------------------------------------------------------------------------')
subprocess.run([
    python_executable, "./model/model_train.py",
    "--config", "./model/configs/model_config.yaml",
    "--data", "./model/data/building_heating_load.csv",
    "--models-dir", "./model/pickles",
])
print('\n')
print('# ----------------------------------------------------------------------------------------------')
print('#                                 2. Run MLflow                                                 ')
print('# ----------------------------------------------------------------------------------------------')
subprocess.run([
    python_executable, "mlflow/run_mlflow.py",
    "--config", "./model/configs/model_config.yaml",
    "--models-dir", "./model/pickles",
    "--mlflow-tracking-uri", "http://localhost:5555"
])
print('\n')
print('# ----------------------------------------------------------------------------------------------')
print('#                                3. Run MLflow Docker Compose                                   ')
print('# ----------------------------------------------------------------------------------------------')
print("Starting docker compose services...")
subprocess.run([
    "docker", "compose", "-f", "./mlflow/compose.yaml", "up", "-d"
], check=True)

# Wait a bit for MLflow to start
import time
print("Waiting for MLflow to initialize...")
time.sleep(2)
print('\n')

print('# ----------------------------------------------------------------------------------------------')
print('#                              4. Run Docker Compose for FastAPI & Streamlit')
print('# ----------------------------------------------------------------------------------------------')
print("Starting docker compose services...")
subprocess.run([
    "docker", "compose", "-f", "compose.yaml", "up", "-d"
], check=True)