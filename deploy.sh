#!/bin/bash

# Accent Detector Deployment Script
# This script helps deploy the Accent Detector application to various environments

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage
function show_usage {
    echo -e "${YELLOW}Usage:${NC} $0 [options]"
    echo -e "${YELLOW}Options:${NC}"
    echo "  --local         Deploy locally using Docker"
    echo "  --aws           Deploy to AWS using Terraform"
    echo "  --heroku        Deploy to Heroku"
    echo "  --gcp           Deploy to Google Cloud Platform"
    echo "  --azure         Deploy to Azure"
    echo "  --k8s           Deploy to Kubernetes"
    echo "  --help          Show this help message"
    exit 1
}

# Function to check if a command exists
function command_exists {
    command -v "$1" >/dev/null 2>&1
}

# Function to check for required environment variables
function check_env_vars {
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set.${NC}"
        echo "Please set it using: export OPENAI_API_KEY=your_api_key"
        exit 1
    fi
}

# Function to deploy locally using Docker
function deploy_local {
    echo -e "${GREEN}Deploying locally using Docker...${NC}"
    
    # Check if Docker is installed
    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed.${NC}"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command_exists docker-compose; then
        echo -e "${RED}Error: docker-compose is not installed.${NC}"
        echo "Please install docker-compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
    fi
    
    # Build and start the containers
    echo "Building and starting containers..."
    docker-compose up -d --build
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "The application is now running at: http://localhost:8501"
}

# Function to deploy to AWS using Terraform
function deploy_aws {
    echo -e "${GREEN}Deploying to AWS using Terraform...${NC}"
    
    # Check if Terraform is installed
    if ! command_exists terraform; then
        echo -e "${RED}Error: Terraform is not installed.${NC}"
        echo "Please install Terraform: https://learn.hashicorp.com/tutorials/terraform/install-cli"
        exit 1
    }
    
    # Check if AWS CLI is installed
    if ! command_exists aws; then
        echo -e "${RED}Error: AWS CLI is not installed.${NC}"
        echo "Please install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
        exit 1
    }
    
    # Check AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}Error: AWS credentials not configured.${NC}"
        echo "Please configure AWS credentials: aws configure"
        exit 1
    }
    
    # Initialize Terraform
    echo "Initializing Terraform..."
    cd terraform
    terraform init
    
    # Apply Terraform configuration
    echo "Applying Terraform configuration..."
    terraform apply -var="openai_api_key=$OPENAI_API_KEY"
    
    # Get outputs
    ECR_REPO=$(terraform output -raw ecr_repository_url)
    LB_DNS=$(terraform output -raw load_balancer_dns)
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "ECR Repository: $ECR_REPO"
    echo "Load Balancer DNS: $LB_DNS"
    echo "The application will be available at: http://$LB_DNS"
    
    # Build and push Docker image
    echo "Building and pushing Docker image..."
    cd ..
    aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
    docker build -t $ECR_REPO:latest .
    docker push $ECR_REPO:latest
    
    echo -e "${GREEN}Docker image pushed successfully!${NC}"
}

# Function to deploy to Heroku
function deploy_heroku {
    echo -e "${GREEN}Deploying to Heroku...${NC}"
    
    # Check if Heroku CLI is installed
    if ! command_exists heroku; then
        echo -e "${RED}Error: Heroku CLI is not installed.${NC}"
        echo "Please install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    }
    
    # Check if logged in to Heroku
    if ! heroku whoami > /dev/null 2>&1; then
        echo "Logging in to Heroku..."
        heroku login
    fi
    
    # Create Heroku app if it doesn't exist
    if ! heroku apps:info accent-detector > /dev/null 2>&1; then
        echo "Creating Heroku app..."
        heroku create accent-detector
    fi
    
    # Add buildpacks
    echo "Adding buildpacks..."
    heroku buildpacks:clear -a accent-detector
    heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git -a accent-detector
    heroku buildpacks:add heroku/python -a accent-detector
    
    # Set environment variables
    echo "Setting environment variables..."
    heroku config:set OPENAI_API_KEY=$OPENAI_API_KEY -a accent-detector
    
    # Deploy to Heroku
    echo "Deploying to Heroku..."
    git push heroku main
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "The application is now running at: https://accent-detector.herokuapp.com"
}

# Function to deploy to Google Cloud Platform
function deploy_gcp {
    echo -e "${GREEN}Deploying to Google Cloud Platform...${NC}"
    
    # Check if gcloud is installed
    if ! command_exists gcloud; then
        echo -e "${RED}Error: Google Cloud SDK is not installed.${NC}"
        echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
        exit 1
    }
    
    # Check if logged in to GCP
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null 2>&1; then
        echo "Logging in to Google Cloud..."
        gcloud auth login
    fi
    
    # Get project ID
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: No Google Cloud project selected.${NC}"
        echo "Please select a project: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    }
    
    # Build and push Docker image
    echo "Building and pushing Docker image..."
    gcloud builds submit --tag gcr.io/$PROJECT_ID/accent-detector
    
    # Deploy to Cloud Run
    echo "Deploying to Cloud Run..."
    gcloud run deploy accent-detector \
        --image gcr.io/$PROJECT_ID/accent-detector \
        --platform managed \
        --allow-unauthenticated \
        --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
    
    # Get the URL
    SERVICE_URL=$(gcloud run services describe accent-detector --platform managed --format="value(status.url)")
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "The application is now running at: $SERVICE_URL"
}

# Function to deploy to Azure
function deploy_azure {
    echo -e "${GREEN}Deploying to Azure...${NC}"
    
    # Check if Azure CLI is installed
    if ! command_exists az; then
        echo -e "${RED}Error: Azure CLI is not installed.${NC}"
        echo "Please install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    }
    
    # Check if logged in to Azure
    if ! az account show > /dev/null 2>&1; then
        echo "Logging in to Azure..."
        az login
    fi
    
    # Create resource group
    echo "Creating resource group..."
    az group create --name accent-detector-rg --location eastus
    
    # Create Azure Container Registry
    echo "Creating Azure Container Registry..."
    az acr create --resource-group accent-detector-rg --name accentdetectoracr --sku Basic
    
    # Log in to ACR
    echo "Logging in to ACR..."
    az acr login --name accentdetectoracr
    
    # Build and push Docker image
    echo "Building and pushing Docker image..."
    docker build -t accentdetectoracr.azurecr.io/accent-detector:latest .
    docker push accentdetectoracr.azurecr.io/accent-detector:latest
    
    # Create container instance
    echo "Creating container instance..."
    az container create \
        --resource-group accent-detector-rg \
        --name accent-detector \
        --image accentdetectoracr.azurecr.io/accent-detector:latest \
        --dns-name-label accent-detector \
        --ports 8501 \
        --environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
    
    # Get the FQDN
    FQDN=$(az container show --resource-group accent-detector-rg --name accent-detector --query ipAddress.fqdn --output tsv)
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "The application is now running at: http://$FQDN:8501"
}

# Function to deploy to Kubernetes
function deploy_k8s {
    echo -e "${GREEN}Deploying to Kubernetes...${NC}"
    
    # Check if kubectl is installed
    if ! command_exists kubectl; then
        echo -e "${RED}Error: kubectl is not installed.${NC}"
        echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
        exit 1
    }
    
    # Check if connected to a Kubernetes cluster
    if ! kubectl cluster-info > /dev/null 2>&1; then
        echo -e "${RED}Error: Not connected to a Kubernetes cluster.${NC}"
        echo "Please configure kubectl to connect to your cluster."
        exit 1
    }
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace accent-detector > /dev/null 2>&1; then
        echo "Creating namespace..."
        kubectl create namespace accent-detector
    fi
    
    # Create secret for OpenAI API key
    echo "Creating secret for OpenAI API key..."
    kubectl create secret generic accent-detector-secrets \
        --from-literal=openai-api-key=$OPENAI_API_KEY \
        --namespace accent-detector \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    echo "Applying Kubernetes manifests..."
    kubectl apply -f kubernetes/deployment.yaml --namespace accent-detector
    
    # Wait for deployment to be ready
    echo "Waiting for deployment to be ready..."
    kubectl rollout status deployment/accent-detector --namespace accent-detector
    
    # Get the service URL
    SERVICE_IP=$(kubectl get service accent-detector-service --namespace accent-detector -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    echo -e "${GREEN}Deployment successful!${NC}"
    echo "The application is now running at: http://$SERVICE_IP"
}

# Main script logic
if [ $# -eq 0 ]; then
    show_usage
fi

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --local)
            check_env_vars
            deploy_local
            ;;
        --aws)
            check_env_vars
            deploy_aws
            ;;
        --heroku)
            check_env_vars
            deploy_heroku
            ;;
        --gcp)
            check_env_vars
            deploy_gcp
            ;;
        --azure)
            check_env_vars
            deploy_azure
            ;;
        --k8s)
            check_env_vars
            deploy_k8s
            ;;
        --help)
            show_usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_usage
            ;;
    esac
    shift
done
