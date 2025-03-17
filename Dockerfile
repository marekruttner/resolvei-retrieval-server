# Use Python 3.12.3 as the base
FROM python:3.12.3-slim

# Prevent Python from writing .pyc files / buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first and install (taking advantage of Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Now copy the rest of the code
COPY . /app

# Expose port 8000 (documentation only; Docker Compose can handle the port mapping)
EXPOSE 8000

# Set the default command
CMD ["uvicorn", "frontend:app", "--host", "0.0.0.0", "--port", "8000"]
