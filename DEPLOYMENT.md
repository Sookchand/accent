# Deployment Guide for English Accent Detector

This document provides detailed instructions for deploying the English Accent Detector application to various environments.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
   - [Streamlit Cloud](#streamlit-cloud)
   - [Heroku](#heroku)
   - [AWS](#aws)
   - [Google Cloud Platform](#google-cloud-platform)
   - [Azure](#azure)
4. [Environment Variables](#environment-variables)
5. [Troubleshooting](#troubleshooting)

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- OpenAI API key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/accent-detector.git
   cd accent-detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your OpenAI API key.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Access the application at http://localhost:8501

## Docker Deployment

### Prerequisites

- Docker installed on your system
- OpenAI API key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/accent-detector.git
   cd accent-detector
   ```

2. Create a `.env` file with your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your OpenAI API key.

3. Build the Docker image:
   ```bash
   docker build -t accent-detector .
   ```

4. Run the Docker container:
   ```bash
   docker run -p 8501:8501 --env-file .env accent-detector
   ```

5. Access the application at http://localhost:8501

## Cloud Deployment

### Streamlit Cloud

Streamlit Cloud is the easiest way to deploy Streamlit applications.

1. Push your code to a GitHub repository.

2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud).

3. Create a new app and connect it to your GitHub repository.

4. Add your OpenAI API key as a secret in the Streamlit Cloud dashboard:
   - Go to "Advanced settings" > "Secrets"
   - Add your API key in the format:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. Deploy the app.

### Heroku

1. Create a `Procfile` in the root directory:
   ```
   web: streamlit run app.py
   ```

2. Create a `runtime.txt` file:
   ```
   python-3.9.7
   ```

3. Install the Heroku CLI and log in:
   ```bash
   heroku login
   ```

4. Create a new Heroku app:
   ```bash
   heroku create accent-detector
   ```

5. Add the FFmpeg buildpack:
   ```bash
   heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
   heroku buildpacks:add heroku/python
   ```

6. Set the OpenAI API key as an environment variable:
   ```bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key_here
   ```

7. Deploy the app:
   ```bash
   git push heroku main
   ```

8. Open the app:
   ```bash
   heroku open
   ```

### AWS

#### Using Elastic Beanstalk

1. Install the AWS CLI and EB CLI:
   ```bash
   pip install awscli awsebcli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Initialize EB application:
   ```bash
   eb init -p python-3.8 accent-detector
   ```

4. Create a `.ebextensions` directory and add a configuration file `01_packages.config`:
   ```yaml
   packages:
     yum:
       ffmpeg: []
   ```

5. Create a `Procfile`:
   ```
   web: streamlit run app.py
   ```

6. Create the environment:
   ```bash
   eb create accent-detector-env
   ```

7. Set environment variables:
   ```bash
   eb setenv OPENAI_API_KEY=your_openai_api_key_here
   ```

8. Deploy the application:
   ```bash
   eb deploy
   ```

#### Using ECS with Docker

1. Create an ECR repository:
   ```bash
   aws ecr create-repository --repository-name accent-detector
   ```

2. Authenticate Docker to ECR:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
   ```

3. Build and tag the Docker image:
   ```bash
   docker build -t accent-detector .
   docker tag accent-detector:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/accent-detector:latest
   ```

4. Push the image to ECR:
   ```bash
   docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/accent-detector:latest
   ```

5. Create an ECS cluster, task definition, and service using the AWS Management Console or AWS CLI.

### Google Cloud Platform

#### Using Cloud Run

1. Install the Google Cloud SDK and initialize:
   ```bash
   gcloud init
   ```

2. Build and push the Docker image to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/<your-project-id>/accent-detector
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy accent-detector \
     --image gcr.io/<your-project-id>/accent-detector \
     --platform managed \
     --allow-unauthenticated \
     --set-env-vars OPENAI_API_KEY=your_openai_api_key_here
   ```

### Azure

#### Using Azure Container Instances

1. Install the Azure CLI and log in:
   ```bash
   az login
   ```

2. Create a resource group:
   ```bash
   az group create --name accent-detector-rg --location eastus
   ```

3. Create an Azure Container Registry:
   ```bash
   az acr create --resource-group accent-detector-rg --name accentdetectoracr --sku Basic
   ```

4. Log in to the registry:
   ```bash
   az acr login --name accentdetectoracr
   ```

5. Build and push the Docker image:
   ```bash
   docker build -t accentdetectoracr.azurecr.io/accent-detector:latest .
   docker push accentdetectoracr.azurecr.io/accent-detector:latest
   ```

6. Create a container instance:
   ```bash
   az container create \
     --resource-group accent-detector-rg \
     --name accent-detector \
     --image accentdetectoracr.azurecr.io/accent-detector:latest \
     --dns-name-label accent-detector \
     --ports 8501 \
     --environment-variables OPENAI_API_KEY=your_openai_api_key_here
   ```

## Environment Variables

The application uses the following environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for Whisper transcription | Yes |

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed on your system or included in your deployment environment.

   - Ubuntu/Debian: `apt-get install ffmpeg`
   - CentOS/RHEL: `yum install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

2. **OpenAI API key issues**: Verify that your API key is correctly set in the environment variables.

3. **Memory issues**: If you encounter memory errors when processing large videos, consider:
   - Limiting the video duration
   - Processing only a portion of the audio
   - Increasing the memory allocation in your deployment environment

4. **Port conflicts**: If port 8501 is already in use, you can specify a different port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the [Streamlit documentation](https://docs.streamlit.io/)
2. Check the [OpenAI API documentation](https://platform.openai.com/docs/api-reference)
3. Open an issue in the GitHub repository

## Security Considerations

1. Never commit your `.env` file or expose your API keys in your code.
2. Use HTTPS for all production deployments.
3. Consider implementing authentication for your application in production.
4. Regularly update dependencies to patch security vulnerabilities.
