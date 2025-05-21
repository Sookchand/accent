variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "openai_api_key" {
  description = "OpenAI API key for Whisper transcription"
  type        = string
  sensitive   = true
}
