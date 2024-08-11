# Use the official Python base image
FROM python:3.7-slim

# Install OpenMP runtime
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
        
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Specify the command to run the model training script
CMD ["python", "DataModelling.py"]
