variable "project_id" {
  description = "GCP project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for the network, Cloud SQL, Artifact Registry and the model bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the cluster. L4 GPUs (g2 machines) must be offered in this zone."
  type        = string
  default     = "us-central1-a"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "bellwether"
}

variable "general_machine_type" {
  description = "Machine type for gateway, analyzer and system pods."
  type        = string
  default     = "e2-standard-4"
}

variable "gpu_max_nodes" {
  description = "Upper bound for the L4 node pool; it scales to zero when idle."
  type        = number
  default     = 3
}

variable "db_tier" {
  description = "Cloud SQL machine tier for the analyzer's PostgreSQL."
  type        = string
  default     = "db-custom-1-3840"
}

variable "deletion_protection" {
  description = "Protect the cluster and database from terraform destroy."
  type        = bool
  default     = true
}
