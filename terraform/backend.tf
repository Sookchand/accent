terraform {
  backend "s3" {
    bucket         = "accent-detector-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "accent-detector-terraform-locks"
  }
}
