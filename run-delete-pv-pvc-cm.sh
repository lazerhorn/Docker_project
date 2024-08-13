#!/bin/bash

# Delete all PersistentVolumeClaims in the namespace
echo "Deleting all PersistentVolumeClaims..."
kubectl delete pvc --all

# Delete all PersistentVolumes in the namespace
echo "Deleting all PersistentVolumes..."
kubectl delete pv --all

# Delete all ConfigMaps in the namespace
echo "Deleting all ConfigMaps..."
kubectl delete configmap --all

echo "All resources have been deleted successfully."