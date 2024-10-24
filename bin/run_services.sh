#!/bin/bash

# TODO: Use this file in your Dockerfile to run the services

# Build the Docker image
docker build -t abalone-fastapi-app -f Dockerfile.app .

# Run the Docker container
docker run -d -p 8000:8000 abalone-fastapi-app
