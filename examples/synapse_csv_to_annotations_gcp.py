#!/usr/bin/env python3
"""Example demonstrating synapse Parquet processing to line annotations on Google Cloud.

This script reads synapse data from a Parquet file stored in GCS and converts it to
Neuroglancer line annotations using a Dask cluster running on Google Cloud Platform.

Requirements:
    - dask-cloudprovider (install with: pip install dask-cloudprovider[gcp])
    - Google Cloud credentials configured (gcloud auth application-default login)
    - Appropriate GCP permissions to create Compute Engine instances
    - Parquet file accessible from GCS (gs://bucket/path/to/file.parquet)
    
Python Environment Setup:
    The default daskdev/dask:latest Docker image does NOT include Neuroglancer or
    TensorStore. You have two options:
    
    1. Use --gcp-extra-bootstrap to install packages at startup:
       --gcp-extra-bootstrap "pip install neuroglancer tensorstore gcsfs pyarrow"
       
    2. Build a custom Docker image with all dependencies pre-installed (recommended
       for production). Then use --gcp-docker-image to specify it.
       
    Note: TensorStore is required for writing annotations. If using extra_bootstrap,
    the first run will be slower as packages are installed.

Network Connectivity:
    IMPORTANT: This script must be run from within the GCP network (e.g., Cloud Shell
    or a VM in the same project) OR you must configure firewall rules to allow access
    to ports 8786-8787 from your IP address.
    
    The scheduler listens on an internal IP by default. To connect from outside GCP:
    1. Run from Cloud Shell: gcloud cloud-shell ssh
    2. Or run from a VM in the same project
    3. Or configure firewall rules and ensure public_ingress=True (default)
"""

import logging
import os
import sys
import shutil
import argparse
from typing import Any
import time

# Configure Dask distributed logging to suppress INFO messages from workers
# This must be done before importing dask.distributed or creating a cluster
import dask
#dask.config.set({'logging.distributed': 'error'})

import neuroglancer
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Set logging level to INFO to see progress messages
logging.basicConfig(level=logging.INFO)

from precomputed_python.distributed_annotations.pipeline import AnnotationPipeline
import neuroglancer.static_file_server

def create_synapse_annotation_bag(
    parquet_path: str,
    num_rows: int | None = None,
    npartitions: int | None = None,
    dask_client: Any = None,
) -> Any:  # Returns dask.bag.Bag
    """Create a distributed Dask bag from synapse Parquet file.
    
    Workers read Parquet row groups directly from GCS - no main process collection.
    Parquet supports random access, so true distributed reading is possible.
    
    Args:
        parquet_path: Path to Parquet file (gs://, s3://, or local path)
        num_rows: Number of rows to process (None = all). For testing, use a small number.
                  Note: This computes the head locally (for testing only).
        npartitions: Number of partitions. If None, uses reasonable default.
        dask_client: Dask client (used to determine number of workers if npartitions is None)
        
    Returns:
        Dask bag of (id, geometry, properties, relationships) tuples
        - id: synapse ID (bigint)
        - geometry: [pre_x, pre_y, pre_z, post_x, post_y, post_z] (line annotation)
        - properties: {"size": size_value}
        - relationships: [[pre_pt_root_id], [post_pt_root_id]]
    """
    import dask.dataframe as dd
    import dask.bag as db
    
    # Determine number of partitions if not provided
    if npartitions is None:
        if dask_client is not None:
            try:
                n_workers = len(dask_client.scheduler_info()['workers'])
                # Use 2 partitions per worker for good parallelism
                npartitions = n_workers * 2
            except (KeyError, AttributeError):
                npartitions = 10  # Default fallback
        else:
            npartitions = 10  # Default fallback
    
    print(f"Reading Parquet from {parquet_path}...")
    print(f"  Partitions: {npartitions}")
    if num_rows:
        print(f"  Limiting to first {num_rows:,} rows for testing")
    
    # Read Parquet into Dask DataFrame
    # Parquet supports random access and partial reads, so distributed reading works well
    # For GCS, authentication uses Google Application Default Credentials
    # (automatically available on GCP VMs via service account)
    df = dd.read_parquet(
        parquet_path,
        engine='pyarrow',  # PyArrow is the default and works well with Dask
        # Parquet files store schema, so dtype specification not needed
    )
    
    # Define required columns
    required_columns = [
        'id',
        'pre_pt_position_x', 'pre_pt_position_y', 'pre_pt_position_z',
        'post_pt_position_x', 'post_pt_position_y', 'post_pt_position_z',
        'size',
        'pre_pt_root_id', 'post_pt_root_id',
    ]
    
    # Verify required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Parquet file: {missing_columns}")
    
    # Select only required columns to save memory
    df = df[required_columns]
    
    # Limit rows if specified (for testing)
    # Note: For testing, we limit the DataFrame before converting to bag
    # For production, remove this limit to process all rows
    if num_rows:
        # head() on Dask DataFrame computes and returns a pandas DataFrame
        # Convert back to Dask DataFrame to keep it as a Dask DataFrame
        df_pandas = df.head(num_rows)
        df = dd.from_pandas(df_pandas, npartitions=min(npartitions, len(df_pandas)))
    
    # Create mapping from column name to index (now in known order)
    col_idx = {name: idx for idx, name in enumerate(required_columns)}
    
    # Convert DataFrame to bag - each row becomes a tuple
    # to_bag(index=False) converts each row to a tuple without the index
    # Tuple order matches required_columns order
    bag = df.to_bag(index=False, format='tuple')
    
    # Transform bag tuples to annotation format
    # We use column indices to extract values from the tuple
    def row_to_annotation(row_tuple):
        """Convert a DataFrame row tuple to annotation tuple format.
        
        Args:
            row_tuple: Tuple from to_bag() containing row values in column order
            
        Returns:
            Tuple (id, geometry, properties, relationships)
        """
        # Extract fields by column name (using precomputed indices)
        ann_id = int(row_tuple[col_idx['id']])
        
        # Line geometry: [pre_x, pre_y, pre_z, post_x, post_y, post_z]
        geometry = [
            float(row_tuple[col_idx['pre_pt_position_x']]),
            float(row_tuple[col_idx['pre_pt_position_y']]),
            float(row_tuple[col_idx['pre_pt_position_z']]),
            float(row_tuple[col_idx['post_pt_position_x']]),
            float(row_tuple[col_idx['post_pt_position_y']]),
            float(row_tuple[col_idx['post_pt_position_z']]),
        ]
        
        # Properties: size
        properties = {
            'size': float(row_tuple[col_idx['size']]),
        }
        
        # Relationships: two relationship types
        # Format: list of lists, one list per relationship type
        relationships = [
            [int(row_tuple[col_idx['pre_pt_root_id']])],
            [int(row_tuple[col_idx['post_pt_root_id']])],
        ]
        
        return (ann_id, geometry, properties, relationships)
    
    # Transform the bag
    bag = bag.map(row_to_annotation)
    
    return bag


def get_gcp_project() -> str | None:
    """Get GCP project from gcloud configuration."""
    try:
        import subprocess
        result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project'],
            capture_output=True,
            text=True,
            check=True,
        )
        project = result.stdout.strip()
        return project if project else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process synapse Parquet file to Neuroglancer line annotations on Google Cloud"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path (local or gs://, s3://)",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to Parquet file (gs://, s3://, or local path). For GCP, use gs:// path.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows to process (None = all). For testing, use a small number.",
    )
    
    # GCP cluster configuration
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help="GCP project ID. If not provided, uses gcloud default project.",
    )
    parser.add_argument(
        "--gcp-region",
        type=str,
        default="us-east1",
        help="GCP region for cluster (default: us-east1)",
    )
    parser.add_argument(
        "--gcp-zone",
        type=str,
        default=None,
        help="GCP zone for cluster. If not provided, uses a zone in the region.",
    )
    parser.add_argument(
        "--gcp-credentials",
        type=str,
        default=None,
        help="Path to GCP service account JSON file. If not provided, uses Application Default Credentials.",
    )
    
    # Cluster size configuration
    parser.add_argument(
        "--gcp-n-workers",
        type=int,
        default=2,
        help="Number of worker nodes (default: 2)",
    )
    parser.add_argument(
        "--gcp-machine-type",
        type=str,
        default="n1-standard-4",
        help="GCP machine type for workers (default: n1-standard-4 = 4 vCPU, 15GB RAM). "
             "Examples: n1-standard-4, n1-standard-8, n1-highmem-4, n1-highmem-8",
    )
    parser.add_argument(
        "--gcp-worker-disk-size",
        type=int,
        default=50,
        help="Disk size in GB for worker nodes (default: 50)",
    )
    parser.add_argument(
        "--gcp-worker-disk-type",
        type=str,
        default="pd-standard",
        help="Disk type for worker nodes (default: pd-standard). Options: pd-standard, pd-ssd",
    )
    parser.add_argument(
        "--gcp-docker-image",
        type=str,
        default=None,
        help="Custom Docker image to use for workers and scheduler. "
             "If not specified, uses daskdev/dask:latest (which requires extra_bootstrap). "
             "Recommended: Build a custom image with Neuroglancer and TensorStore pre-installed.",
    )
    parser.add_argument(
        "--gcp-extra-bootstrap",
        type=str,
        default=None,
        help="Extra bootstrap commands to run on workers (e.g., pip install commands). "
             "Example: 'pip install neuroglancer tensorstore gcsfs'",
    )
    
    # Dask worker configuration
    parser.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=4,
        help="Threads per worker (default: 4, should match number of CPUs in machine type)",
    )
    parser.add_argument(
        "--dask-memory-limit",
        type=str,
        default=None,
        help="Memory limit per worker (e.g., '14GB'). Default: auto (machine RAM - 1GB)",
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--spatial-limit",
        type=int,
        default=10000,
        help="Spatial index limit per cell (default: 10000)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Skip annotation processing and only open the viewer (assumes annotations already exist)",
    )
    
    # Add Neuroglancer viewer arguments
    import neuroglancer.cli
    neuroglancer.cli.add_server_arguments(parser)
    
    args = parser.parse_args()
    
    # Handle Neuroglancer server arguments
    neuroglancer.cli.handle_server_arguments(args)
    
    # Determine output path and whether it's local
    is_local_path = not (args.output_path.startswith("gs://") or args.output_path.startswith("s3://"))
    if is_local_path:
        output_path = os.path.abspath(args.output_path)
    else:
        output_path = args.output_path
    
    # Skip processing if view-only mode
    if args.view_only:
        print("\n" + "=" * 60)
        print("VIEW-ONLY MODE: Skipping annotation processing")
        print("=" * 60)
        print(f"  Output path: {output_path}")
        print(f"  Assuming annotations already exist at this location")
        print()
    else:
        # Clean up existing output directory if it's a local path
        if is_local_path and os.path.exists(output_path):
            print(f"\nCleaning up existing output directory: {output_path}")
            try:
                shutil.rmtree(output_path)
                print(f"  Removed existing directory")
            except Exception as e:
                print(f"  WARNING: Failed to remove existing directory: {e}")
                print(f"  Continuing anyway...")
            print()
    
    # Skip Dask setup and processing if view-only mode
    if not args.view_only:
        # Check for dask-cloudprovider
        try:
            from dask_cloudprovider.gcp import GCPCluster
            from dask.distributed import Client
        except ImportError:
            print("ERROR: dask-cloudprovider is not installed.")
            print("Install with: pip install dask-cloudprovider[gcp]")
            sys.exit(1)
        
        # Determine GCP project
        gcp_project = args.gcp_project
        if gcp_project is None:
            gcp_project = get_gcp_project()
            if gcp_project is None:
                print("ERROR: GCP project not specified and gcloud default project not found.")
                print("Please set --gcp-project or run: gcloud config set project PROJECT_ID")
                sys.exit(1)
        
        # Set up credentials if provided
        if args.gcp_credentials:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.gcp_credentials
            print(f"Using GCP credentials from: {args.gcp_credentials}")
        
        print(f"\nGCP Configuration:")
        print(f"  Project: {gcp_project}")
        if args.gcp_zone:
            print(f"  Zone: {args.gcp_zone}")
        else:
            print(f"  Region: {args.gcp_region} (will use first zone in region)")
        print()
        
        print(f"Dask Cluster Configuration:")
        print(f"  Workers: {args.gcp_n_workers}")
        print(f"  Machine type: {args.gcp_machine_type}")
        print(f"  Disk size: {args.gcp_worker_disk_size} GB ({args.gcp_worker_disk_type})")
        print(f"  Threads per worker: {args.dask_threads_per_worker}")
        if args.dask_memory_limit:
            print(f"  Memory limit per worker: {args.dask_memory_limit}")
        else:
            print(f"  Memory limit per worker: auto (machine RAM - 1GB)")
        print()
        
        # Determine memory limit
        memory_limit = args.dask_memory_limit
        if memory_limit is None:
            # Default: machine RAM - 1GB for OS
            # For n1-standard-4, that's ~15GB - 1GB = 14GB
            memory_limit = "auto"
        
        # Create GCP cluster
        print("Creating GCP Dask cluster...")
        print("  This may take a few minutes to start VMs...")
        
        # Determine zone (required by GCPCluster, not region)
        if args.gcp_zone:
            gcp_zone = args.gcp_zone
        else:
            # Use a zone in the specified region (default to first zone)
            # For simplicity, just use the region + "-a" as default
            gcp_zone = f"{args.gcp_region}-b"
            print(f"  Note: Using zone {gcp_zone} (first zone in region {args.gcp_region})")
        
        cluster_kwargs = {
            'projectid': gcp_project,
            'zone': gcp_zone,
            'n_workers': args.gcp_n_workers,
            'machine_type': args.gcp_machine_type,
            'filesystem_size': args.gcp_worker_disk_size,  # Note: filesystem_size, not disk_size
            'disk_type': args.gcp_worker_disk_type,
            # Explicitly set GPU parameters to 0 (int) to avoid config issues
            'scheduler_ngpus': 0,
            'worker_ngpus': 0,
            'ngpus': 0
        }
        
        # Configure Docker image (if custom image specified)
        if args.gcp_docker_image:
            cluster_kwargs['docker_image'] = args.gcp_docker_image
            print(f"  Using custom Docker image: {args.gcp_docker_image}")
        else:
            print("  Using default Docker image: daskdev/dask:latest")
            if not args.gcp_extra_bootstrap:
                print("  WARNING: Default image does not include Neuroglancer or TensorStore!")
                print("  Specify --gcp-extra-bootstrap or --gcp-docker-image to install dependencies.")
        
        # Configure bootstrap commands (if specified)
        # extra_bootstrap should be a list of commands
        if args.gcp_extra_bootstrap:
            cluster_kwargs['extra_bootstrap'] = [args.gcp_extra_bootstrap]
            print(f"  Extra bootstrap commands: {args.gcp_extra_bootstrap}")
        
        print("  Creating GCPCluster (this may take a few minutes for VMs to start)...")
        cluster = GCPCluster(**cluster_kwargs)
        
        print("  GCPCluster created")
        print("  Waiting for scheduler to be ready...")
        print("  Note: This may take 2-5 minutes for the VM to start and scheduler to initialize")
        
        # Poll for scheduler address with progress updates
        import sys
        import time as time_module
        scheduler_timeout = 600  # 10 minutes
        start_time = time_module.time()
        last_print = 0
        
        while True:
            try:
                scheduler_address = getattr(cluster, 'scheduler_address', None)
                if scheduler_address:
                    print(f"  Scheduler address found: {scheduler_address}")
                    break
            except Exception:
                pass  # scheduler_address might not be available yet
            
            elapsed = time_module.time() - start_time
            if elapsed > scheduler_timeout:
                print(f"\n  ERROR: Scheduler did not become ready within {scheduler_timeout}s")
                print(f"  Check scheduler VM in GCP Console")
                print(f"  SSH to scheduler VM and run: sudo docker ps -a")
                print(f"  Then check logs: sudo docker logs <container-id>")
                raise TimeoutError("Scheduler did not become ready")
            
            # Print progress every 15 seconds
            if elapsed - last_print >= 15:
                print(f"  Still waiting for scheduler... ({int(elapsed)}s elapsed)")
                sys.stdout.flush()
                last_print = elapsed
            
            time_module.sleep(2)
        
        print("  Connecting to cluster...")
        sys.stdout.flush()
        
        try:
            # Client() will connect to the scheduler
            client = Client(cluster, timeout=120)  # 2 minute timeout for connection
        except Exception as e:
            print(f"\n  ERROR: Failed to connect to cluster: {e}")
            print(f"  Scheduler address: {scheduler_address}")
            print(f"  Troubleshooting:")
            print(f"    1. Check scheduler VM logs: sudo docker logs <scheduler-container>")
            print(f"    2. Verify firewall rules allow traffic on ports 8786-8787")
            print(f"    3. Check if scheduler is accessible: curl http://{scheduler_address.split('://')[1].split(':')[0]}:8787/status")
            raise
        
        print(f"  Cluster connection established")
        print(f"  Dashboard: {client.dashboard_link}")
        print()
        
        try:
            # Check scheduler status first
            print("Checking scheduler status...")
            try:
                scheduler_info = client.scheduler_info()
                print(f"  Scheduler is running")
            except Exception as e:
                print(f"  WARNING: Could not get scheduler info: {e}")
                print(f"  The scheduler may still be starting up...")
            
            # Wait for workers to be ready
            print(f"\nWaiting for {args.gcp_n_workers} workers to be ready...")
            print("  This may take several minutes if bootstrap commands are running...")
            import time as time_module
            start_time = time_module.time()
            try:
                client.wait_for_workers(args.gcp_n_workers, timeout=600)  # 10 minute timeout for bootstrap
            except Exception as e:
                elapsed = time_module.time() - start_time
                print(f"\n  ERROR: Timed out waiting for workers after {elapsed:.1f}s")
                print(f"  Error: {e}")
                print(f"\n  Troubleshooting:")
                print(f"    1. Check the scheduler VM logs in GCP Console")
                print(f"    2. SSH into the scheduler VM and check Docker logs: docker logs <scheduler-container>")
                print(f"    3. Verify bootstrap commands completed successfully")
                print(f"    4. Check firewall rules allow traffic on ports 8786-8787")
                raise
            
            n_workers_ready = len(client.scheduler_info()['workers'])
            elapsed = time_module.time() - start_time
            print(f"  {n_workers_ready} workers ready (took {elapsed:.1f}s)")
            print()
            
            # Create distributed annotation bag from Parquet
            print(f"Creating distributed annotation bag from Parquet...")
            annotation_bag = create_synapse_annotation_bag(
                parquet_path=args.parquet_path,
                num_rows=args.num_rows,
                npartitions=None,  # Auto-determine based on number of workers
                dask_client=client,
            )
            
            print(f"  Annotations will be read on workers (distributed, no main process collection)")
            print()
            
            # Set up pipeline with Dask enabled
            print("Initializing annotation pipeline with Dask support...")
            pipeline = AnnotationPipeline(
                output_path=output_path,
                coordinate_space=neuroglancer.CoordinateSpace(
                    names=["x", "y", "z"],
                    units=["nm", "nm", "nm"],
                    scales=[4.0, 4.0, 40.0],  # 4nm voxel size (typical for EM data)
                ),
                annotation_type="line",  # Line annotations (pre -> post)
                properties=[
                    neuroglancer.AnnotationPropertySpec(
                        id="size",
                        type="float32",
                        description="Synapse size",
                    ),
                ],
                relationships=["pre_pt_root_id", "post_pt_root_id"],  # Two relationship types
                spatial_index_config={
                    "limit": args.spatial_limit,
                },
                sharding_config={
                    "by_id": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "num_shards": 16,
                    },
                    "spatial": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "max_annotations_per_shard": 100000,
                    },
                    "relationships": {
                        "pre_pt_root_id": {
                            "@type": "neuroglancer_uint64_sharded_v1",
                            "num_shards": 16,
                        },
                        "post_pt_root_id": {
                            "@type": "neuroglancer_uint64_sharded_v1",
                            "num_shards": 16,
                        },
                    },
                },
                use_dask=True,
                dask_client=client,
            )
            print("  Pipeline initialized with Dask support")
            print()
            
            # Process annotations
            print("Processing annotations with Dask...")
            print("=" * 60)
            pipeline_start = time.time()
            
            pipeline.process(annotation_bag=annotation_bag)
            
            pipeline_time = time.time() - pipeline_start
            print("=" * 60)
            print()
            
            num_processed = args.num_rows if args.num_rows else "all"
            print(f"Completed in {pipeline_time:.2f} seconds")
            if args.num_rows:
                throughput = args.num_rows / pipeline_time if pipeline_time > 0 else 0
                print(f"  Throughput: {throughput:,.0f} annotations/second")
            print()
            
            print(f"Annotations written to: {output_path}")
            print()
        
        finally:
            print("Shutting down GCP cluster...")
            print("  This may take a minute to terminate VMs...")
            client.close()
            cluster.close()
            print("  Cluster shut down")
            print()
    
    # Set up Neuroglancer viewer
    print("Setting up Neuroglancer viewer...")
    
    # Set up coordinate space for viewer
    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[4.0, 4.0, 40.0],
    )
    
    # Create viewer
    viewer = neuroglancer.Viewer()
    
    # Set up state
    s = viewer.state
    
    # Set up annotation layer
    annotation_layer = neuroglancer.AnnotationLayer(
        source=output_path,
    )
    s.layers["synapses"] = annotation_layer
    
    # Set up image layer
    s.layers["image"] = neuroglancer.ImageLayer(
        source="precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
    )
    
    # Set up segmentation layer
    s.layers["segmentation"] = neuroglancer.SegmentationLayer(
        source="graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1",
    )
    
    # Link annotation layer relationships to segmentation layer
    annotation_layer.linked_segmentation_layer = {
        "pre_pt_root_id": "segmentation",
        "post_pt_root_id": "segmentation",
    }
    
    # Set up local file server if output is local
    if is_local_path:
        static_server = neuroglancer.static_file_server.StaticFileServer(output_path)
        annotation_layer.source = static_server.base_url
        print(f"  Local file server: {static_server.base_url}")
    else:
        annotation_layer.source = output_path
        print(f"  Remote path: {output_path}")
    
    # Set layout and view settings
    s.layout = "xy-3d"
    s.show_slices = False
    
    # Set initial position and zoom (center of typical EM volume)
    # These are reasonable defaults, but may need adjustment based on data
    s.position = coordinate_space.voxel_coordinates_to_nm([512, 512, 256])
    s.cross_section_scale = 0.5
    s.projection_scale = 1000
    
    print()
    print("=" * 60)
    print("Neuroglancer viewer ready!")
    print(f"  URL: {viewer}")
    print("=" * 60)
    print()
    
    # Keep viewer open
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

