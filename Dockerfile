# Use Python 3.11 slim image (CHANGED FROM 3.10)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . /app

# Expose the port your Flask app uses
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["python", "app.py"]

