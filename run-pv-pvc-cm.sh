# Apply PersistentVolumes
kubectl apply -f k8s/pv/pv-training-data.yml
kubectl apply -f k8s/pv/pv-validation-data.yml
kubectl apply -f k8s/pv/pv-inference-rf.yml
kubectl apply -f k8s/pv/pv-inference-xgb.yml
kubectl apply -f k8s/pv/pv-model-rf.yml
kubectl apply -f k8s/pv/pv-model-xgb.yml


# Apply PersistentVolumeClaims
kubectl apply -f k8s/pvc/pvc-training-data.yml
kubectl apply -f k8s/pvc/pvc-validation-data.yml
kubectl apply -f k8s/pvc/pvc-inference-rf.yml
kubectl apply -f k8s/pvc/pvc-inference-xgb.yml
kubectl apply -f k8s/pvc/pvc-model-rf.yml
kubectl apply -f k8s/pvc/pvc-model-xgb.yml


kubectl apply -f k8s/configmaps/config.yml