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
trap cleanup EXIT INT TERM

# Function to wait for Pods to be running
wait_for_pod_ready() {
    local job_name="$1"
    local pod_name

    echo "Waiting for Pods to be running..."
    for (( i=0; i<300; i+=10 )); do
        pod_name=$(kubectl get pods --selector=job-name="${job_name}" --output=jsonpath='{.items[0].metadata.name}')

        if [ -n "$pod_name" ]; then
            pod_status=$(kubectl get pod "$pod_name" --output=jsonpath='{.status.phase}')
            
            if [ "$pod_status" == "Running" ]; then
                print_info "$GREEN" "Pod ${pod_name} is running."
                return
            fi
        fi
        
        sleep 1
    done

    print_info "$RED" "Timeout waiting for Pod ${pod_name} to be running."
    exit 1
}

# Function to stream logs from a Pod associated with a job
stream_pod_logs() {
    local job_name="$1"
    local pod_name

    while true; do
        pod_name=$(kubectl get pods --selector=job-name="${job_name}" --output=jsonpath='{.items[0].metadata.name}')
        
        if [ -n "$pod_name" ]; then
            print_info "$CYAN" "Streaming logs for pod ${pod_name}..."
            kubectl logs -f "${pod_name}"
            break
        else
            print_info "$RED" "No pods found for job ${job_name}. Retrying..."
            sleep 1
        fi
    done
}

# Run data-processing-job
print_info "$CYAN" "Applying data-processing-job..."
kubectl apply -f k8s/jobs/data-processing-job.yml
wait_for_pod_ready "data-processing-job"
stream_pod_logs "data-processing-job"

# Run data-modelling-job
print_info "$CYAN" "Applying data-modelling-job..."
kubectl apply -f k8s/jobs/data-modelling-job.yml
wait_for_pod_ready "data-modelling-job"
stream_pod_logs "data-modelling-job"

# Run model-inference-job
print_info "$CYAN" "Applying model-inference-job..."
kubectl apply -f k8s/jobs/model-inference-job.yml
wait_for_pod_ready "model-inference-job"
stream_pod_logs "model-inference-job"

print_info "$GREEN" "All jobs have been processed."