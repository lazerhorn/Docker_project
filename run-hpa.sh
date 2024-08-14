# Deploy data-processing-hpa
kubectl apply -f k8s/hpa/data-processing-hpa.yml

# Deploy data-modelling-hpa
kubectl apply -f k8s/hpa/data-modelling-hpa.yml

# Deploy model-inference-hpa
kubectl apply -f k8s/hpa/model-inference-hpa.yml