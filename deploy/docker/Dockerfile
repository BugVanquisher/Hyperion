# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
