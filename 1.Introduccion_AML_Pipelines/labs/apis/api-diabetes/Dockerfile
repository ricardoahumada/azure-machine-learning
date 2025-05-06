# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api-modelo:app"]