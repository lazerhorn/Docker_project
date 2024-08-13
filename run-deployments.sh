#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Define color codes
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print messages in color
print_info() {
    local color="$1"
    shift
    echo -e "${color}$*${NC}"
}

# Define environment variables for mount paths
SOURCE_PATH=${SOURCE_PATH:-/mnt/c/Users/adapt/OneDrive/game/Documents/Desktop/aisol/raw_data}
TARGET_PATH=${TARGET_PATH:-/data/raw}

# Start Minikube mount in the background
print_info "$CYAN" "Starting Minikube mount from ${SOURCE_PATH} to ${TARGET_PATH}..."
minikube mount ${SOURCE_PATH}:${TARGET_PATH} &

# Capture the PID of the Minikube mount process
MOUNT_PID=$!

# Deploy data-processing-deployment
kubectl apply -f k8s/deployments/data-processing-deployment.yml

# Deploy data-modelling-deployment
kubectl apply -f k8s/deployments/data-modelling-deployment.yml

# Deploy model-inference-deployment
kubectl apply -f k8s/deployments/model-inference-deployment.yml

# Wait for user input to exit
print_info "$CYAN" "Press any key to exit..."
read -n 1 -s

# Cleanup Minikube mount
print_info "$CYAN" "Stopping Minikube mount..."
kill $MOUNT_PID
wait $MOUNT_PID 2>/dev/null
print_info "$CYAN" "Minikube mount stopped."
