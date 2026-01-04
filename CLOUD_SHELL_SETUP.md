# Running on Google Cloud Platform

The simplest way to run the script on GCP is to use **Google Cloud Shell**, which provides:
- Pre-authenticated gcloud CLI
- Docker pre-installed
- Network access to GCP resources
- No VM setup required

## Option 1: Google Cloud Shell (Recommended for Quick Runs)

### Steps:

1. **Open Cloud Shell:**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Click the Cloud Shell icon (top right) or visit [shell.cloud.google.com](https://shell.cloud.google.com)

2. **Clone or upload your code:**
   ```bash
   # Option A: If your code is in a git repository
   git clone YOUR_REPO_URL
   cd YOUR_REPO_DIR
   
   # Option B: Upload files using Cloud Shell Editor
   # Click the "Open Editor" button in Cloud Shell, then upload files
   ```

3. **Set your project:**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Run the script:**
   ```bash
   # Make the script executable
   chmod +x run_on_gcp.sh
   
   # Run with your parameters (example with specific paths)
   ./run_on_gcp.sh \
     --parquet-path gs://allen-minnie-phase3/dask_synapses/synapses_pni_2_v1_filtered_view.parquet \
     --output-path gs://allen-minnie-phase3/dask_synapses/synapse_test1 \
     --num-rows 1000000 \
     --gcp-n-workers 5 \
     --gcp-machine-type n1-standard-4
   ```

   Or run the Python script directly:
   ```bash
   python3 examples/synapse_csv_to_annotations_gcp.py \
     --gcp-project $(gcloud config get-value project) \
     --gcp-docker-image caveconnectome/precomputed-python-dask:latest \
     --parquet-path gs://allen-minnie-phase3/dask_synapses/synapses_pni_2_v1_filtered_view.parquet \
     --output-path gs://allen-minnie-phase3/dask_synapses/synapse_test1 \
     --num-rows 1000000 \
     --gcp-n-workers 5 \
     --gcp-machine-type n1-standard-4
   ```

### Cloud Shell Limitations:
- **5GB persistent disk** (limited storage)
- **Session timeout** after 20 minutes of inactivity
- **CPU/memory limits** (1 vCPU, 3.75GB RAM) - fine for running the script, but cluster workers run separately

## Option 2: Compute Engine VM (Recommended for Production)

For more persistent/reusable setups, create a small VM:

### Steps:

1. **Create a VM:**
   ```bash
   # Create a small VM with Docker pre-installed
   gcloud compute instances create dask-runner \
     --project=YOUR_PROJECT_ID \
     --zone=us-east1-b \
     --machine-type=e2-standard-2 \
     --image-family=cos-stable \
     --image-project=cos-cloud \
     --boot-disk-size=20GB \
     --scopes=https://www.googleapis.com/auth/cloud-platform
   ```

   Or use a Ubuntu image if you prefer:
   ```bash
   gcloud compute instances create dask-runner \
     --project=YOUR_PROJECT_ID \
     --zone=us-east1-b \
     --machine-type=e2-standard-2 \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=20GB \
     --scopes=https://www.googleapis.com/auth/cloud-platform
   ```

2. **SSH into the VM:**
   ```bash
   gcloud compute ssh dask-runner --zone=us-east1-b
   ```

3. **Install dependencies:**
   ```bash
   # For Container-Optimized OS (cos-stable):
   # Docker is pre-installed, but you need to install Python
   # Consider using the Ubuntu image instead for easier setup
   
   # For Ubuntu:
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip git docker.io
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   # Log out and back in for docker group to take effect
   ```

4. **Clone your code and run:**
   ```bash
   git clone YOUR_REPO_URL
   cd YOUR_REPO_DIR
   
   # Install Python dependencies (if needed)
   pip3 install dask-cloudprovider[gcp] neuroglancer
   
   # Run the script
   python3 examples/synapse_csv_to_annotations_gcp.py \
     --gcp-project $(gcloud config get-value project) \
     --gcp-docker-image caveconnectome/precomputed-python-dask:latest \
     --parquet-path gs://allen-minnie-phase3/dask_synapses/synapses_pni_2_v1_filtered_view.parquet \
     --output-path gs://allen-minnie-phase3/dask_synapses/synapse_test1 \
     --num-rows 1000000 \
     --gcp-n-workers 5
   ```

5. **Clean up when done:**
   ```bash
   gcloud compute instances delete dask-runner --zone=us-east1-b
   ```

## Option 3: Using the Helper Script

A helper script is provided (`run_on_gcp.sh`) that detects if you're running on GCP and configures accordingly:

```bash
# From Cloud Shell or a GCP VM
chmod +x run_on_gcp.sh
./run_on_gcp.sh \
  --parquet-path gs://allen-minnie-phase3/dask_synapses/synapses_pni_2_v1_filtered_view.parquet \
  --output-path gs://allen-minnie-phase3/dask_synapses/synapse_test1 \
  --num-rows 1000000
```

## Authentication

### Cloud Shell
- Already authenticated - no setup needed!

### Compute Engine VM
- Uses the VM's service account (automatically configured)
- Ensure the service account has these roles:
  - `Compute Instance Admin` (to create/manage VMs)
  - `Service Account User` (to use service accounts)
  - Or use `--scopes=https://www.googleapis.com/auth/cloud-platform` for full access

## Network Configuration

When running from within GCP (Cloud Shell or VM):
- The script automatically uses internal networking
- Workers communicate via internal IPs
- More secure and faster

When running from outside GCP:
- Requires `public_ingress=True` (default in dask-cloudprovider)
- Workers expose public IPs
- Requires firewall rules (usually auto-configured)

## Troubleshooting

### "Permission denied" errors
- Ensure your account/VM has the necessary IAM roles
- Check service account permissions

### "Connection timeout" errors
- Ensure you're running from within GCP (Cloud Shell or VM)
- Check firewall rules if using public networking

### Docker image not found
- Ensure the image is pushed to Docker Hub or GCR
- Check the image name is correct
- For GCR: `gcloud auth configure-docker`

## Cost Considerations

- **Cloud Shell**: Free (but limited resources)
- **Compute Engine VM**: ~$0.10-0.20/hour for e2-standard-2
- **Dask Cluster Workers**: Pay per worker instance (e.g., 5x n1-standard-4 = ~$0.19/hour per worker)
- **Storage**: Pay for GCS usage

Remember to delete VMs and clusters when done!

