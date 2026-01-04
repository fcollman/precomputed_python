#!/usr/bin/env python3
"""Example demonstrating synapse Parquet processing to line annotations.

This script reads synapse data from a Parquet file and converts it to
Neuroglancer line annotations using the distributed annotation framework.
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
dask.config.set({'logging.distributed': 'error'})

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
    
    Workers read Parquet row groups directly - no main process collection.
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
    # For public GCS buckets, no credentials needed
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


def main():
    parser = argparse.ArgumentParser(
        description="Process synapse Parquet file to Neuroglancer line annotations"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output_synapse_annotations",
        help="Output path for annotations (local directory or gs://, s3:// path)",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=os.path.expanduser("~/Downloads/synapses_pni_2_v1_filtered_view.parquet"),
        help="Path to Parquet file (gs://, s3://, or local path). "
             "Parquet supports distributed reading (random access).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows to process (for testing). Default: all rows",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=None,
        help="Number of Dask workers (default: CPU count)",
    )
    parser.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=1,
        help="Threads per Dask worker (default: 1)",
    )
    parser.add_argument(
        "--dask-memory-limit",
        type=str,
        default=None,
        help="Memory limit per worker (e.g., '2GB'). Default: 'auto'",
    )
    parser.add_argument(
        "--spatial-limit",
        type=int,
        default=10000,
        help="Spatial index limit per cell (default: 1000)",
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
        # Set up Dask client
        try:
            import dask
            import dask.bag as db
            print(f"Dask version: {dask.__version__}")
        except ImportError:
            print("ERROR: Dask is not installed. Install with: pip install dask")
            sys.exit(1)
        
        from dask.distributed import Client, LocalCluster
        
        if args.dask_workers is None:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.dask_workers
        
        # Determine memory limit per worker
        if args.dask_memory_limit is None:
            memory_limit = "auto"
            memory_limit_str = "auto (system memory / workers)"
        else:
            memory_limit = args.dask_memory_limit
            memory_limit_str = memory_limit
        
        print(f"\nSetting up Dask cluster with {num_workers} workers...")
        print(f"  Threads per worker: {args.dask_threads_per_worker}")
        print(f"  Memory limit per worker: {memory_limit_str}")
        
        # Create local cluster
        cluster = LocalCluster(
            processes=True,
            n_workers=num_workers,
            threads_per_worker=args.dask_threads_per_worker,
            memory_limit=memory_limit,
            silence_logs=logging.WARNING,
        )
        client = Client(cluster)
        
        print(f"  Dashboard: {client.dashboard_link}")
        print()
        
        try:
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
            # Clean up Dask client and cluster
            print("Shutting down Dask cluster...")
            client.close()
            cluster.close()
    
    # Launch Neuroglancer viewer
    print("=" * 60)
    print("Launching Neuroglancer viewer...")
    print("=" * 60)
    print()
    
    
    
    # Create viewer
    viewer = neuroglancer.Viewer()
    
    # Set up source URL for local paths
    server = None
    if is_local_path:
        # Create static file server for local paths
        
        server = neuroglancer.static_file_server.StaticFileServer(
            static_dir=output_path,
            bind_address=args.bind_address or "127.0.0.1",
            daemon=True,
        )
        annotation_source = f"precomputed://{server.url}"
    else:
        # For cloud storage (gs:// or s3://), use the URL directly
        annotation_source = f"precomputed://{output_path.rstrip('/')}"
    
    with viewer.txn() as s:
        # Add image layer
        s.layers["image"] = neuroglancer.ImageLayer(
            source="precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
        )
        
        # Add segmentation layer
        s.layers["segmentation"] = neuroglancer.SegmentationLayer(
            source="graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1",
        )
        
        # Add annotation layer
        annotation_layer = neuroglancer.AnnotationLayer(
            source=annotation_source,
        )
        
        # Link relationships to segmentation layer
        annotation_layer.linked_segmentation_layer = {
            "pre_pt_root_id": "segmentation",
            "post_pt_root_id": "segmentation",
        }
        
        s.layers["synapses"] = annotation_layer
        
        # Set initial view (centered on origin with reasonable zoom)
        # Coordinates are in nm, so use reasonable initial position
        #s.position = [50000.0, 50000.0, 50000.0]  # nm
        #s.cross_section_scale = 1000.0  # Reasonable zoom for EM data
        #s.projection_scale = 100000.0
        s.layout = "xy-3d"
        s.show_slices = False
    
    print()
    print("Neuroglancer viewer launched!")
    print(f"  Image layer: precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em")
    print(f"  Segmentation layer: graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1")
    print(f"  Annotation source: {annotation_source}")
    print(f"  Annotation type: line (pre -> post synapses)")
    print(f"  Relationships: pre_pt_root_id, post_pt_root_id (linked to segmentation layer)")
    if server:
        print(f"  Static file server: {server.url}")
    print()
    print("Viewer URL:")
    print(viewer)


if __name__ == "__main__":
    main()

