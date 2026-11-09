locals {
  services = [
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "servicenetworking.googleapis.com",
    "sqladmin.googleapis.com",
  ]
  workload_pool = "${var.project_id}.svc.id.goog"
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------- network

resource "google_compute_network" "vpc" {
  name                    = "${var.name}-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.enabled]
}

resource "google_compute_subnetwork" "nodes" {
  name                     = "${var.name}-nodes"
  region                   = var.region
  network                  = google_compute_network.vpc.id
  ip_cidr_range            = "10.10.0.0/20"
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/14"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

# ---------------------------------------------------------------- cluster

resource "google_container_cluster" "main" {
  name                     = var.name
  location                 = var.zone
  network                  = google_compute_network.vpc.id
  subnetwork               = google_compute_subnetwork.nodes.id
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = var.deletion_protection

  # Dataplane V2 (eBPF) enforces the NetworkPolicies in deploy/k8s/base.
  datapath_provider = "ADVANCED_DATAPATH"

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = local.workload_pool
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
}

resource "google_container_node_pool" "general" {
  name     = "general"
  cluster  = google_container_cluster.main.id
  location = var.zone

  autoscaling {
    min_node_count = 2
    max_node_count = 6
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.general_machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot = true
    }
  }
}

resource "google_container_node_pool" "l4" {
  name     = "l4"
  cluster  = google_container_cluster.main.id
  location = var.zone

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "g2-standard-8"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    labels       = { workload = "inference" }

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot = true
    }
  }
}

# ---------------------------------------------------------------- images and models

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.name
  format        = "DOCKER"
  depends_on    = [google_project_service.enabled]
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-${var.name}-models"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = !var.deletion_protection

  versioning {
    enabled = true
  }
}

# ---------------------------------------------------------------- identities (Workload Identity)

resource "google_service_account" "triton" {
  account_id   = "${var.name}-triton"
  display_name = "Triton model server (reads the model repository)"
}

resource "google_storage_bucket_iam_member" "triton_reads_models" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.triton.email}"
}

resource "google_service_account_iam_member" "triton_wi" {
  service_account_id = google_service_account.triton.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${local.workload_pool}[bellwether/triton]"
}

resource "google_service_account" "analyzer" {
  account_id   = "${var.name}-analyzer"
  display_name = "Bellwether analyzer"
}

resource "google_project_iam_member" "analyzer_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.analyzer.email}"
}

resource "google_service_account_iam_member" "analyzer_wi" {
  service_account_id = google_service_account.analyzer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${local.workload_pool}[bellwether/bellwether-analyzer]"
}

# ---------------------------------------------------------------- Cloud SQL (private IP)

resource "google_compute_global_address" "sql_peering" {
  name          = "${var.name}-sql-peering"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "sql" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.sql_peering.name]
}

resource "google_sql_database_instance" "analyzer" {
  name                = "${var.name}-analyzer"
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = var.deletion_protection
  depends_on          = [google_service_networking_connection.sql]

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }
  }
}

resource "google_sql_database" "analyzer" {
  name     = "bellwether"
  instance = google_sql_database_instance.analyzer.name
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "google_sql_user" "analyzer" {
  name     = "bellwether"
  instance = google_sql_database_instance.analyzer.name
  password = random_password.db.result
}
