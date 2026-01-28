# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for scanning
RUN apt-get update && apt-get install -y iputils-ping sudo

# Upgrade pip and configure trusted hosts for corporate networks with SSL inspection
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

COPY requirements.txt .

# Install requirements with extended timeout, retries, and trusted hosts for SSL bypass
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Install PyTorch separately with extended timeout and trusted hosts
RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]