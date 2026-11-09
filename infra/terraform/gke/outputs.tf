output "cluster_name" {
  value = google_container_cluster.main.name
}

output "get_credentials" {
  description = "Configure kubectl for the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --zone ${var.zone} --project ${var.project_id}"
}

output "image_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "model_bucket" {
  value = google_storage_bucket.models.name
}

output "analyzer_db_url" {
  description = "Store as the bellwether-db secret: kubectl -n bellwether create secret generic bellwether-db --from-literal=url=..."
  value       = "postgresql+psycopg2://${google_sql_user.analyzer.name}:${random_password.db.result}@${google_sql_database_instance.analyzer.private_ip_address}:5432/${google_sql_database.analyzer.name}"
  sensitive   = true
}
