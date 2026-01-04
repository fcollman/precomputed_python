#!/usr/bin/env python3
"""Example demonstrating distributed annotation processing with Dask.

This example shows how to use the distributed annotation framework with Dask
for efficient parallel processing and memory management of large-scale datasets.
"""

import logging
import os
import sys
import shutil
from typing import Any
import time
# Configure Dask distributed logging to suppress INFO messages from workers
# This must be done before importing dask.distributed or creating a cluster
import dask
dask.config.set({'logging.distributed': 'error'})

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Set logging level to INFO to see progress messages
logging.basicConfig(level=logging.INFO)

from precomputed_python.distributed_annotations.encoder import AnnotationEncoder, EncodedAnnotation
from precomputed_python.distributed_annotations.pipeline import AnnotationPipeline
from precomputed_python.distributed_annotations.writer import DistributedAnnotationWriter


def generate_annotation_partition(
    partition_info: tuple[int, int, int, int],  # (partition_idx, annotations_per_partition, start_id, n_related_ids)
) -> list[tuple]:
    """Generate annotations for a specific partition - runs on worker.
    
    This function generates annotations for a single partition independently on a worker,
    avoiding collection of data through the main process.
    
    Args:
        partition_info: Tuple of (partition_idx, annotations_per_partition, start_id, n_related_ids)
        
    Returns:
        List of (id, geometry, properties, relationships) tuples
    """
    import random
    
    partition_idx, annotations_per_partition, start_id, n_related_ids = partition_info
    
    # Use partition index as seed for reproducibility (optional)
    # Each partition gets its own random sequence
    random.seed(42 + partition_idx)
    
    annotations = []
    
    for i in range(annotations_per_partition):
        # Calculate global annotation ID
        ann_id = start_id + i
        
        # Random 3D position
        point = [
            random.uniform(0, 100000),
            random.uniform(0, 100000),
            random.uniform(0, 100000),
        ]
        
        # Random color property
        color = [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ]
        
        # Randomly assign to one of the related object IDs (if relationships enabled)
        # Relationships format: list of lists, one list per relationship type
        # Each list contains the related object IDs for that relationship type
        if n_related_ids > 0:
            related_id = random.randint(1, n_related_ids)
            relationships = [[related_id]]  # Single relationship type with one related ID
        else:
            relationships = []
        
        # Append tuple: (id, geometry, properties, relationships)
        annotations.append((ann_id, point, {"color": color}, relationships))
    
    return annotations


def create_distributed_annotation_bag(
    num_annotations: int,
    n_related_ids: int,
    npartitions: int | None = None,
    dask_client: Any = None,
) -> Any:  # Returns dask.bag.Bag
    """Create a distributed Dask bag where each partition generates its own annotations.
    
    This pattern allows workers to generate annotations independently, avoiding
    collection of data through the main process.
    
    Args:
        num_annotations: Total number of annotations to generate
        n_related_ids: Number of related object IDs to choose from (0 means no relationships)
        npartitions: Number of partitions. If None, uses a reasonable default
        dask_client: Dask client (used to determine number of workers if npartitions is None)
        
    Returns:
        Dask bag of (id, geometry, properties, relationships) tuples
    """
    import dask.bag as db
    
    # Determine number of partitions
    if npartitions is None:
        if dask_client is not None:
            try:
                n_workers = len(dask_client.scheduler_info()['workers'])
                npartitions = max(1, n_workers * 2)  # 2 partitions per worker
            except (KeyError, AttributeError):
                npartitions = 8  # Default fallback
        else:
            npartitions = 8  # Default if no client provided
    
    # Calculate annotations per partition
    annotations_per_partition = (num_annotations + npartitions - 1) // npartitions  # Ceiling division
    
    # Create partition info tuples: (partition_idx, annotations_per_partition, start_id, n_related_ids)
    partition_infos = []
    for partition_idx in range(npartitions):
        start_id = partition_idx * annotations_per_partition
        # Last partition might have fewer annotations
        if partition_idx == npartitions - 1:
            actual_count = num_annotations - start_id
        else:
            actual_count = annotations_per_partition
        partition_infos.append((partition_idx, actual_count, start_id, n_related_ids))
    
    # Create bag from partition info tuples - each worker will generate its partition
    bag = db.from_sequence(partition_infos, npartitions=npartitions)
    
    # Map: Each worker generates its partition's annotations independently
    bag = bag.map(generate_annotation_partition).flatten()
    
    return bag


def main():
    """Main function demonstrating distributed annotation processing."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Test distributed annotation processing with Dask"
    )
    parser.add_argument(
        "--num-annotations",
        type=int,
        default=100000,
        help="Number of annotations to generate (default: 100000)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.abspath("./output_annotations_dask"),
        help="Output path for annotations (default: ./output_annotations_dask)",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=None,
        help="Number of Dask workers (default: auto-detect CPU count)",
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
        help="Memory limit per worker (e.g., '2GB', '1GB'). Default: 'auto' (divides total memory by number of workers). For annotation processing, 1-2GB per worker is usually sufficient.",
    )
    parser.add_argument(
        "--spatial-limit",
        type=int,
        default=1000,
        help="Spatial index limit per cell (default: 1000)",
    )
    parser.add_argument(
        "--n-related-ids",
        type=int,
        default=100,
        help="Number of related object IDs to assign annotations to (0 = no relationships, default: 0)",
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
    
    # Handle Neuroglancer server arguments (needs to be after args.parse_args())
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
    
    # Handle Neuroglancer server arguments (needs to be after args.parse_args())
    neuroglancer.cli.handle_server_arguments(args)
    
    # Skip Dask setup and processing if view-only mode
    if not args.view_only:
        # Check if Dask is available
        try:
            import dask
            import dask.bag as db
            print(f"Dask version: {dask.__version__}")
        except ImportError:
            print("ERROR: Dask is not installed. Install with: pip install dask")
            sys.exit(1)
        
        # Set up Dask client
        from dask.distributed import Client, LocalCluster
        import dask.distributed as dd
        
        if args.dask_workers is None:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.dask_workers
        
        # Determine memory limit per worker
        if args.dask_memory_limit is None:
            # Default to "auto" which divides total system memory by number of workers
            # This is conservative but may limit worker count on systems with limited RAM
            memory_limit = "auto"
            memory_limit_str = "auto (system memory / workers)"
        else:
            # User specified explicit memory limit
            memory_limit = args.dask_memory_limit
            memory_limit_str = memory_limit
    
        print(f"\nSetting up Dask cluster with {num_workers} workers...")
        print(f"  Threads per worker: {args.dask_threads_per_worker}")
        print(f"  Memory limit per worker: {memory_limit_str}")
        print(f"  Note: For annotation processing, 1-2GB per worker is usually sufficient")
        
        # Create local cluster explicitly to have full control over worker count
        # Using LocalCluster explicitly ensures we can set n_workers regardless of CPU count
        cluster = LocalCluster(
            processes=True,
            n_workers=num_workers,
            threads_per_worker=args.dask_threads_per_worker,
            memory_limit=memory_limit,
            silence_logs=logging.WARNING,  # Reduce cluster startup noise
        )
        client = Client(cluster)
  
        
        print(f"  Dashboard: {client.dashboard_link}")
        print()
        
        try:
            # Create distributed annotation bag
            # Each partition will generate its annotations independently on workers
            print(f"Creating distributed annotation bag for {args.num_annotations:,} annotations...")
            if args.n_related_ids > 0:
                print(f"  Relationships enabled: each annotation will be assigned to one of {args.n_related_ids} related object IDs")
            
            annotation_bag = create_distributed_annotation_bag(
                num_annotations=args.num_annotations,
                n_related_ids=args.n_related_ids,
                npartitions=None,  # Auto-determine based on number of workers
                dask_client=client,
            )
            
            print(f"  Annotations will be generated on workers (distributed, no main process collection)")
            print()
            
            # Set up pipeline with Dask enabled
            import neuroglancer
            
            print("Initializing annotation pipeline with Dask support...")
            pipeline = AnnotationPipeline(
                output_path=output_path,
                coordinate_space=neuroglancer.CoordinateSpace(
                    names=["x", "y", "z"],
                    units=["nm", "nm", "nm"],
                    scales=[1.0, 1.0, 1.0],
                ),
                annotation_type="point",
                properties=[
                    neuroglancer.AnnotationPropertySpec(
                        id="color",
                        type="rgb",
                        description="Annotation color",
                    ),
                ],
                relationships=["segment"] if args.n_related_ids > 0 else [],
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
                        "segment": {
                            "@type": "neuroglancer_uint64_sharded_v1",
                            "num_shards": 16,
                        }
                    } if args.n_related_ids > 0 else {},
                },
                use_dask=True,  # Enable Dask for distributed processing
                dask_client=client,  # Pass the Dask client
            )
            print("  Pipeline initialized with Dask support")
            print()
            
            # Process annotations using Dask
            print("Processing annotations with Dask...")
            print("=" * 60)
            
            import time
            start_time = time.time()
            
            # Process with pipeline using distributed bag
            # The pipeline will:
            # 1. Encode annotations (distributed - each worker encodes its partition)
            # 2. Build spatial index (distributed)
            # 3. Build relationship indices (distributed)
            # 4. Write all indices (distributed - workers write directly)
            
            # Enable profiling if requested
            profile_file = os.environ.get("DASK_PROFILE_FILE", None)
            if profile_file:
                from dask.distributed import performance_report
                logging.info(f"Profiling enabled, writing report to {profile_file}")
                with performance_report(filename=profile_file):
                    pipeline.process(annotation_bag=annotation_bag)
            else:
                pipeline.process(annotation_bag=annotation_bag)
            
            elapsed_time = time.time() - start_time
            
            print("=" * 60)
            print(f"\nCompleted in {elapsed_time:.2f} seconds")
            print(f"  Throughput: {args.num_annotations / elapsed_time:,.0f} annotations/second")
            print()
            
            # Show Dask task graph info
            print("Dask cluster info:")
            print(f"  Workers: {len(client.scheduler_info()['workers'])}")
            print(f"  Total memory: {sum(w['memory_limit'] for w in client.scheduler_info()['workers'].values()) / 1e9:.2f} GB")
            print()
            
            # Show profiling info
            if profile_file:
                print(f"Performance profile saved to: {profile_file}")
                print("  Open in browser to view detailed task execution timeline")
                print()
            

            
        finally:
            # Clean up Dask client and cluster
            print("Shutting down Dask cluster...")
            client.close()
            cluster.close()

    # Write segment properties file if relationships are enabled (only if not view-only)
    if not args.view_only and args.n_related_ids > 0:
        print("=" * 60)
        print("Writing segment properties file...")
        print("=" * 60)
        
        import tensorstore as ts
        import json
        import os
        
        # Create segment properties directory path
        if is_local_path:
            segprops_path = os.path.join(output_path, "segproperties")
            os.makedirs(segprops_path, exist_ok=True)
            segprops_kvstore_path = segprops_path
        else:
            # For cloud storage, construct the path
            if not output_path.endswith("/"):
                segprops_kvstore_path = output_path + "/segproperties/"
            else:
                segprops_kvstore_path = output_path + "segproperties/"
        
        # Open kvstore for segment properties directory (using same logic as writer)
        if segprops_kvstore_path.startswith("gs://") or segprops_kvstore_path.startswith("s3://"):
            if not segprops_kvstore_path.endswith("/"):
                segprops_kvstore_path = segprops_kvstore_path + "/"
            segprops_kvstore = ts.KvStore.open(segprops_kvstore_path).result()
        elif segprops_kvstore_path.startswith("file://"):
            if not segprops_kvstore_path.endswith("/"):
                segprops_kvstore_path = segprops_kvstore_path + "/"
            segprops_kvstore = ts.KvStore.open(segprops_kvstore_path).result()
        else:
            # Local file path
            abs_path = os.path.abspath(segprops_kvstore_path)
            if not abs_path.endswith(os.sep):
                abs_path = abs_path + os.sep
            segprops_kvstore = ts.KvStore.open({"driver": "file", "path": abs_path}).result()
        
        # Create segment properties JSON
        # IDs are from 1 to n_related_ids
        segment_ids = [str(i) for i in range(1, args.n_related_ids + 1)]
        
        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": segment_ids,
                "properties": []
            }
        }
        
        # Write the info file
        info_json = json.dumps(segment_properties, indent=2)
        segprops_kvstore.write("info", info_json.encode("utf-8")).result()
        
        print(f"  Written segment properties for {args.n_related_ids} segment IDs")
        print(f"  Location: {segprops_kvstore_path}info")
        print()
        
        # Launch Neuroglancer viewer to inspect the written annotations
    print("=" * 60)
    print("Launching Neuroglancer viewer...")
    print("=" * 60)
    
    viewer = neuroglancer.Viewer()
    server = None
    
    if is_local_path:
        # For local paths, use StaticFileServer
        import neuroglancer.static_file_server
        server = neuroglancer.static_file_server.StaticFileServer(
            static_dir=output_path,
            bind_address=args.bind_address or "127.0.0.1",
            daemon=True,
        )
        annotation_source = f"precomputed://{server.url}"
        # Segment properties path for viewer (relative to server root)
        segprops_source = f"precomputed://{server.url}/segproperties"
    else:
        # For cloud storage (gs:// or s3://), use the URL directly
        # Ensure the URL doesn't end with a slash (neuroglancer expects this)
        annotation_source = f"precomputed://{output_path.rstrip('/')}"
        # Segment properties path for viewer
        if not output_path.endswith("/"):
            segprops_source = f"precomputed://{output_path}/segproperties"
        else:
            segprops_source = f"precomputed://{output_path.rstrip('/')}/segproperties"
    
    with viewer.txn() as s:
        # Add segmentation layer if relationships are enabled
        if args.n_related_ids > 0:
            # Create an empty LocalVolume with bounds matching the annotation area
            # Annotations span 0 to 100000 in each dimension (nm), with scale 1.0
            # Create an empty volume (all zeros) that spans the annotation bounds
            # This helps Neuroglancer calculate proper zoom levels
            import numpy as np
            
            # Create empty volume spanning the annotation area (0 to 100000)
            # Using a reasonable size to represent the bounds without being too large
            # The volume is all zeros (no segment data), just for establishing bounds
            volume_size = 1001  # 1001 voxels gives bounds 0-1000, but we'll scale it
            empty_seg_data = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint64)
            
            # Use the same coordinate space as annotations
            dimensions = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                units=["nm", "nm", "nm"],
                scales=[100.0, 100.0, 100.0],
            )
            
            # Create empty LocalVolume positioned at origin (0,0,0)
            # This will give bounds from [0,0,0] to [1000,1000,1000]
            # While not the full 0-100000 range, this provides a reasonable reference
            # for zoom calculations. The segment properties provide the actual segment data.
            empty_volume = neuroglancer.LocalVolume(
                data=empty_seg_data,
                dimensions=dimensions,
                volume_type="segmentation",
            )
            
            # Create segmentation layer with two sources:
            # 1. Segment properties (for segment metadata)
            # 2. Empty LocalVolume (for coordinate space bounds)
            seg_layer = neuroglancer.SegmentationLayer(
                source=[
                    segprops_source,  # Segment properties
                    empty_volume,     # Empty volume for bounds
                ]
            )
            s.layers["segments"] = seg_layer
        
        annotation_layer = neuroglancer.AnnotationLayer(
            source=annotation_source,
            tab="rendering",
            shader="""
void main() {
setColor(prop_color());
}
            """,
        )
        
        # Link annotation layer to segmentation layer if relationships are enabled
        if args.n_related_ids > 0:
            annotation_layer.linked_segmentation_layer = {"segment": "segments"}
        
        s.layers["annotations"] = annotation_layer
        
        # Set initial position to center of the coordinate space
        # Use the bounds from the coordinate space (0 to 100000 based on generation)
        s.position = [500, 500, 500]
        s.layout = "xy-3d"
        s.show_slices = False
        # Set reasonable zoom levels
        # Scale values represent voxel scale factor - larger = more zoomed in
        # For coordinates in nm with scale 1.0, a scale of ~5000 shows about 50 units of data
        # which is a reasonable starting view for annotation visualization
        #s.cross_section_scale = 250.0  # Cross-section (2D) zoom
        #s.projection_scale = 75000.0     # Perspective/3D zoom
    
    print(f"\nNeuroglancer viewer launched!")
    print(f"  Annotation source: {annotation_source}")
    if args.n_related_ids > 0:
        print(f"  Segmentation layer: segments (with {args.n_related_ids} segment IDs)")
        print(f"  Annotation layer linked to segmentation layer via 'segment' relationship")
    if server:
        print(f"  Static file server: {server.url}")
    print(f"\nViewer URL:")
    print(viewer)
    print()
    
if __name__ == "__main__":
    main()

