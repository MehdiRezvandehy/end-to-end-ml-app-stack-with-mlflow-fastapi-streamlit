FROM python:3.11-slim

WORKDIR /app

# Copy ONLY FastAPI app
COPY fastapi/ /app/

# Copy requirements file (must be inside fastapi folder OR project root)
COPY fastapi/requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# Copy model files from project root
COPY ../model/pickles/*.pkl /app/model/pickles/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
