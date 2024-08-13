#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print messages in color
print_info() {
    local color="$1"
    shift
    echo -e "${color}$*${NC}"
}

# Define environment variables for mount paths (Change source according to file path)
SOURCE_PATH=${SOURCE_PATH:-/mnt/c/Users/adapt/OneDrive/game/Documents/Desktop/aisol/raw_data}
TARGET_PATH=${TARGET_PATH:-/data/raw}

# Start Minikube mount in the background
print_info "$CYAN" "Starting Minikube mount from ${SOURCE_PATH} to ${TARGET_PATH}..."
minikube mount ${SOURCE_PATH}:${TARGET_PATH} &

# Capture the PID of the background mount process
MOUNT_PID=$!

# Function to stop Minikube mount and delete all jobs
cleanup() {
    print_info "$YELLOW" "Deleting all jobs..."
    kubectl delete jobs --all --ignore-not-found
    print_info "$GREEN" "All jobs deleted."
    
    print_info "$YELLOW" "Stopping Minikube mount..."
    # Check if the Minikube mount process is still running
    if ps -p $MOUNT_PID > /dev/null; then
        kill $MOUNT_PID
        wait $MOUNT_PID 2>/dev/null
        print_info "$GREEN" "Minikube mount stopped."
    else
        print_info "$RED" "Minikube mount process not found."
    fi
}

# Ensure cleanup is done on exit or interruption
# This ensures the cleanup function is called when the script exits, or when it receives interrupt or terminate signals
trap cleanup EXIT INT TERM

# Function to get logs from the latest pod of a given job
get_pod_logs() {
    local job_name="$1"
    local pod_name
    pod_name=$(kubectl get pods --selector=job-name="${job_name}" --output=jsonpath='{.items[-1].metadata.name}')
    if [ -z "$pod_name" ]; then
        print_info "$RED" "No pods found for job ${job_name}."
    else
        print_info "$CYAN" "Fetching logs for pod ${pod_name}..."
        kubectl logs "${pod_name}"
    fi
}

# Run data-processing-job
print_info "$CYAN" "Applying data-processing-job..."
kubectl apply -f k8s/jobs/data-processing-job.yml
print_info "$CYAN" "Waiting for data-processing-job to complete..."
kubectl wait --for=condition=complete job/data-processing-job --timeout=30m
print_info "$GREEN" "Data-processing-job completed."

# Fetch logs for data-processing-job
get_pod_logs "data-processing-job"

# Run data-modelling-job
print_info "$CYAN" "Applying data-modelling-job..."
kubectl apply -f k8s/jobs/data-modelling-job.yml
print_info "$CYAN" "Waiting for data-modelling-job to complete..."
kubectl wait --for=condition=complete job/data-modelling-job --timeout=30m
print_info "$GREEN" "Data-modelling-job completed."

# Fetch logs for data-modelling-job
get_pod_logs "data-modelling-job"

# Run model-inference-job
print_info "$CYAN" "Applying model-inference-job..."
kubectl apply -f k8s/jobs/model-inference-job.yml
print_info "$CYAN" "Waiting for model-inference-job to complete..."
kubectl wait --for=condition=complete job/model-inference-job --timeout=30m
print_info "$GREEN" "Model-inference-job completed."

# Fetch logs for model-inference-job
get_pod_logs "model-inference-job"

print_info "$GREEN" "All jobs have been processed."
