output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.accent_detector.repository_url
}

output "load_balancer_dns" {
  description = "The DNS name of the load balancer"
  value       = aws_lb.accent_detector.dns_name
}
