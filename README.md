## **How to Run (If not using sh files (manually))**

Before starting any processing modules, you need to mount your local drive to Minikube. This allows Minikube to access the dataset stored on your local machine. Execute the following command:

```bash
minikube mount /mnt/c/Users/adapt/OneDrive/game/Documents/Desktop/aisol/raw_data:/data/raw
```
(Above shown is an example command)

This mounts the local directory `/mnt/c/Users/adapt/OneDrive/game/Documents/Desktop/aisol/raw_data` (Replace this with your file path to the raw_data) to the
`/data/raw` (DO NOT CHANGE THIS FILE PATH) directory within Minikube, enabling it to read the dataset required for processing.

## **How to Run (If using sh files)**

Inside each of the jobs/deployments sh files there are the SOURCE_PATH= in there same thing just need to change that path to your own local path.

### **Run Scripts**

We provide several `run.sh` scripts for different purposes:

- **`run-compose.sh`**  
  This script is used for running `docker-compose.yml` files. It is primarily intended for testing purposes. Docker Compose files are useful for defining and running multi-container Docker applications, but they are typically used for local testing.

- **`run-deployments.sh`**  
  This script deploys all three containerized modules in one go. It sets up the necessary deployments for data processing, data modelling, and model inference, ensuring that all components are deployed simultaneously. 

- **`run-jobs.sh`**  
  This script is for running Jobs that execute only once. Use it when you need to trigger one-time tasks.

- **`run-pv-pvc-cm.sh`**  
  This script creates all the Persistent Volumes (PV), Persistent Volume Claims (PVC), and ConfigMaps (CM) listed in the script. It is essential for setting up the storage and configuration resources required by the deployments and jobs.

- **`run-delete-pv-pvc-cm.sh`**  
  Use this script to delete all the Persistent Volumes (PV), Persistent Volume Claims (PVC), and ConfigMaps (CM) listed in the script. This is useful for cleaning up resources before a fresh deployment.

## **Kubernetes Jobs and Deployments: Implementation Details**

### **Kubernetes Deployments**
**Scaling**
- **Manual Scaling**: We can adjust the number of replicas in the Deployment using the `kubectl scale` command. For example:
  kubectl scale deployment model-inference-deployment --replicas=4

**Self-Healing**
- **Automatic Replacement**: Kubernetes Deployments automatically replace failed or unresponsive Pods to maintain the desired number of replicas. So it already has built in self healing.

**Rollout/Rollback**
- **Rolling Updates**: Kubernetes performs rolling updates by default, gradually replacing old Pods with new ones. (If we use the apply -f command)
- **Rollback**: We can revert to a previous version if issues arise with an update. Kubernetes maintains a history of ReplicaSets for this purpose. (We set this to 5 so 5 versions it can revert back to)

**Resource Management**
- **Resource Requests and Limits**: We have defined resource requests and limits for each container in the Deployment to ensure appropriate resource allocation. For example giving the most resources to modelling as it requires the most for training.

### **Kubernetes Jobs**
**Scaling**
- **Parallelism**: We’ve configured the Job to run multiple pods concurrently by setting the `parallelism` field. For instance, with `parallelism: 1`, only one pod runs at a time, as demonstrated in our examples.
- **Completions**: We’ve set the `completions` field to define the total number of successful completions required for the Job to be considered complete. For example, with `completions: 1`, the Job is marked complete after one successful execution. (Can be seen from our example)

**Self-Healing**
- **Restart Policy**: We’ve set the `restartPolicy` to `OnFailure`, which ensures that failed pods are retried according to the Job’s configuration.
- **Backoff Limit**: We’ve configured the `backoffLimit` to `4`, specifying the number of retry attempts before the Job is marked as failed. This prevents infinite retries and provides control over the retry behavior.

**Rollout/Rollback**
- **Updating Jobs**: We update the Job’s configuration by modifying it and reapplying it. Note that Kubernetes does not support automatic rollback for Jobs, so we manage rollbacks manually if needed.

**Resource Management**
- **Resource Requests and Limits**: We’ve defined resource requests and limits to ensure the Job receives the necessary resources and is protected from resource contention. For example modelling we gave the most resources to it as it requires most for training.


## **Image Versions Logs**
### **Image for Data Processing***
- **`lazerhorn/data_processing:latestv2`**  
  Used in jobs where the processing ends because the code executed in this image completes its task and terminates. This image is designed for jobs that don’t need to keep running in a loop.

- **`lazerhorn/data_processing:latestv3`**  
  Updated code to be used in deployments that do not end by themselves, as the image here processes data and needs to be used in a continuous loop. If we use `lazerhorn/data_modelling:latestv2` in this deployment, a crash loop backoff will occur because the code will terminate and the deployment will need to remake a container which causes crash loop backoff error. Althought the container will restart itself after the crash its not ideal to be used for real deployment thus v3 is needed.

- **`lazerhorn/data_processing:latestv5`**  
  This version addresses an issue present in `latestv3`, where the global status number was not being reset properly. This issue caused the status to increase indefinitely. The `latestv5` image fixes this problem by correctly resetting the global number, preventing it from exceeding the maximum limit.

### **Image for Data Modelling***
- **`lazerhorn/data_modelling:latestv3`**
Used in jobs where the modelling ends because the code executed in this image completes its task and terminates. 
This image is designed for jobs that don’t need to keep running in a loop.

- **`lazerhorn/data_modelling:latestv4`**
Updated code to be used in deployments that do not end by themselves, as the image here processes data and needs to be used in a continuous loop.

### **Image for Data Inference***
- **`lazerhorn/data_inference:latestv4`**
Used in jobs where the inference ends because the code executed in this image completes its task and terminates. 
This image is designed for jobs that don’t need to keep running in a loop.

- **`lazerhorn/data_inference:latestv6`**
Updated code to be used in deployments that do not end by themselves, as the image here processes data and needs to be used in a continuous loop.