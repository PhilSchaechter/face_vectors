# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install SQLite3
RUN apt-get update && apt-get install -y sqlite3
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*  # Clean up to reduce layer size

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# If your Python application has a specific command to run, you can set it here:
CMD ["python", "process_vfile.py"]

# If you need to expose a port (for web applications), uncomment or add this line:
EXPOSE 5000
