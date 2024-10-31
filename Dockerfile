# Use Python 3.11 slim image as base
FROM python:3.11

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 1978

# Run the application
CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "1978"]