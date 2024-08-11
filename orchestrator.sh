#!/bin/bash

# Build the Docker images
docker-compose build

# Start the data_processing container
docker-compose up -d data_processing
docker-compose logs -f data_processing # Wait until data_processing completes

# Start the data_modelling container
docker-compose up -d data_modelling
docker-compose logs -f data_modelling # Wait until data_modelling completes

# Start the model_inference container
docker-compose up -d model_inference
docker-compose logs -f model_inference # Wait until model_inference completes
