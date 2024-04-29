# Deployments

This folder contains scripts and resources for deploying your API as a SageMaker live endpoint with autoscaling capabilities.

## Files

- **build_and_push.sh**: This bash script builds your API container and pushes it to Amazon Elastic Container Registry (ECR) to make it available for deployment.
## Usage

1. Run the `build_and_push.sh` script to build your API container and push it to your ECR repository. Make sure to provide the necessary configurations and credentials in the script, such as the ECR repository name and AWS credentials.

## Prerequisites

- Docker: Make sure you have Docker installed on your local machine for building the API container.
- AWS CLI: Ensure you have the AWS CLI configured with appropriate credentials to push the container to ECR and deploy the SageMaker endpoint.
