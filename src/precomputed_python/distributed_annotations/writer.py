"""TensorStore-based writer for precomputed annotations to cloud storage."""

import json
import logging
import math
import os
import struct
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import mmh3
import tensorstore as ts

from .encoder import AnnotationEncoder, EncodedAnnotation
from .spatial_index import SpatialIndexLevel

logger = logging.getLogger(__name__)

cells_to_shards_ratio_threshold = 500


def _zorder_compressed(coords, grid_shape):
    """Compute compressed z-order (Morton) code for spatial cell coordinates.
    
    This function is copied from neuroglancer.read_precomputed_annotations to avoid
    dependency on neuroglancer in the distributed_annotations module.
    
    Args:
        coords: Tuple or sequence of cell coordinates
        grid_shape: Tuple or sequence of grid dimensions
        
    Returns:
        int: Compressed Morton code as uint64
    """
    zindex = 0
    output_bit = 0
    for bit in range(32):
        for coord, size in zip(coords, grid_shape):
            coord = int(coord)
            if (size - 1) >> bit:
                zindex |= ((coord >> bit) & 1) << output_bit
                output_bit += 1
    return zindex


def _compute_shard_config_from_max_annotations(
    base_sharding_config: dict[str, Any],
    total_annotations: int,
    max_annotations_per_shard: int,
) -> dict[str, Any]:
    """Compute sharding config from max_annotations_per_shard and total annotations.
    
    Calculates the required number of shards (rounded up to next power of 2) to ensure
    each shard has at most max_annotations_per_shard annotations.
    
    Args:
        base_sharding_config: Base sharding configuration (may contain max_annotations_per_shard)
        total_annotations: Total number of annotations
        max_annotations_per_shard: Maximum annotations per shard (target)
        
    Returns:
        Sharding config with computed shard_bits
    """
    import math
    
    # Calculate required number of shards (rounded up)
    required_shards = max(1, math.ceil(total_annotations / max_annotations_per_shard))
    
    # Round up to next power of 2 (shard_bits requires powers of 2)
    shard_bits = 0
    while (1 << shard_bits) < required_shards:
        shard_bits += 1
    
    # Create config based on base config
    config = base_sharding_config.copy()
    
    # Remove max_annotations_per_shard if present (not part of final config)
    config.pop("max_annotations_per_shard", None)
    
    # Set computed shard_bits (override any existing num_shards or shard_bits)
    if "@type" not in config:
        config["@type"] = "neuroglancer_uint64_sharded_v1"
    
    config["shard_bits"] = shard_bits
    config.pop("num_shards", None)  # Remove num_shards if present
    
    # Ensure other required fields are set
    config.setdefault("preshift_bits", 0)
    config.setdefault("hash", "murmurhash3_x86_128")
    config.setdefault("minishard_bits", 0)
    config.setdefault("minishard_index_encoding", "gzip")
    config.setdefault("data_encoding", "gzip")
    
    return config


def _normalize_sharding_config_standalone(sharding: dict[str, Any]) -> dict[str, Any]:
    """Standalone function to normalize sharding config (can be pickled for Dask).
    
    Converts simplified format (e.g., {"num_shards": 16}) to full spec format
    with all required fields.
    
    Args:
        sharding: Sharding configuration (may be simplified or full format)
        
    Returns:
        Normalized sharding config with all required fields
    """
    # If already in full format, check @type
    if "@type" in sharding:
        if sharding.get("@type") != "neuroglancer_uint64_sharded_v1":
            raise ValueError(f"Unsupported sharding type: {sharding.get('@type')}")
    else:
        # Simplified format - add @type
        sharding = sharding.copy()
        sharding["@type"] = "neuroglancer_uint64_sharded_v1"
    
    normalized = {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": sharding.get("preshift_bits", 0),
        "hash": sharding.get("hash", "murmurhash3_x86_128"),
        "minishard_bits": sharding.get("minishard_bits", 0),
        "shard_bits": sharding.get("shard_bits", None),
        "minishard_index_encoding": sharding.get("minishard_index_encoding", "gzip"),
        "data_encoding": sharding.get("data_encoding", "gzip"),
    }
    
    # Convert num_shards to shard_bits if provided
    if "num_shards" in sharding:
        if normalized["shard_bits"] is not None:
            raise ValueError("Cannot specify both 'num_shards' and 'shard_bits'")
        num_shards = sharding["num_shards"]
        # Calculate shard_bits: num_shards = 2^shard_bits
        shard_bits = 0
        while (1 << shard_bits) < num_shards:
            shard_bits += 1
        if (1 << shard_bits) != num_shards:
            raise ValueError(f"num_shards must be a power of 2, got {num_shards}")
        normalized["shard_bits"] = shard_bits
    
    # Ensure shard_bits is set
    if normalized["shard_bits"] is None:
        raise ValueError("Must specify either 'num_shards' or 'shard_bits' in sharding config")
    
    return normalized


def _compute_shard_from_chunk_id(chunk_id: int, config: dict) -> int:
    """Compute shard number from a chunk ID using sharding config.
    
    This is a standalone function that can be pickled for Dask.
    
    Args:
        chunk_id: The chunk ID (e.g., annotation ID, morton code, object ID)
        config: Normalized sharding configuration dict
        
    Returns:
        Shard number
    """
    preshift_bits = config.get("preshift_bits", 0)
    hash_func = config.get("hash", "identity")
    minishard_bits = config.get("minishard_bits", 0)
    shard_bits = config.get("shard_bits", 0)
    
    chunk_id_shifted = chunk_id >> preshift_bits
    
    if hash_func == "identity":
        hashed_id = chunk_id_shifted
    elif hash_func == "murmurhash3_x86_128":
        chunk_id_bytes = chunk_id_shifted.to_bytes(8, 'little', signed=False)
        hash_result = mmh3.hash128(chunk_id_bytes, x64arch=False)
        if isinstance(hash_result, tuple):
            hashed_id = hash_result[0]
            if hashed_id < 0:
                hashed_id = hashed_id + (1 << 64)
            hashed_id = hashed_id & ((1 << 64) - 1)
        else:
            hashed_id = hash_result & ((1 << 64) - 1)
    else:
        raise ValueError(f"Unsupported hash function: {hash_func}")
    
    shard_and_minishard = hashed_id & ((1 << (minishard_bits + shard_bits)) - 1)
    shard = (shard_and_minishard >> minishard_bits) & ((1 << shard_bits) - 1)
    return shard


def _compute_shard_number_standalone(ann: EncodedAnnotation, config: dict) -> int:
    """Standalone shard number computation that can be pickled for Dask.
    
    Returns just the shard number (not a tuple).
    This is a module-level function so it can be properly serialized by Dask.
    """
    return _compute_shard_from_chunk_id(ann.id, config)


# Worker-local cache for TensorStore kvstores
# This avoids repeatedly opening the same kvstore on each worker
_worker_kvstore_cache = {}


def _get_cached_kvstore(cache_key: str, factory_func):
    """Get or create a kvstore from worker-local cache.
    
    This function uses a module-level dictionary that persists per worker process,
    allowing us to reuse TensorStore objects across multiple writes.
    
    Args:
        cache_key: Unique key for this kvstore (e.g., "spatial0_gs://bucket/path")
        factory_func: Function that creates the kvstore if not cached
        
    Returns:
        TensorStore kvstore
    """
    if cache_key not in _worker_kvstore_cache:
        _worker_kvstore_cache[cache_key] = factory_func()
    return _worker_kvstore_cache[cache_key]


def _create_base_kvstore_standalone(output_path: str) -> ts.KvStore:
    """Create a base kvstore for the given output path (standalone, can be pickled).
    
    Args:
        output_path: Path to output (gs://, s3://, file://, or local path)
        
    Returns:
        TensorStore kvstore
    """
    if output_path.startswith("gs://") or output_path.startswith("s3://"):
        path = output_path if output_path.endswith("/") else output_path + "/"
        return ts.KvStore.open(path).result()
    elif output_path.startswith("file://"):
        path = output_path if output_path.endswith("/") else output_path + "/"
        return ts.KvStore.open(path).result()
    else:
        abs_path = os.path.abspath(output_path)
        if not abs_path.endswith(os.sep):
            abs_path = abs_path + os.sep
        return ts.KvStore.open({"driver": "file", "path": abs_path}).result()


def _create_sharded_kvstore_standalone(
    output_path: str,
    key: str,
    normalized_sharding: dict[str, Any],
) -> ts.KvStore:
    """Create a sharded kvstore (standalone, can be pickled).
    
    Args:
        output_path: Base output path
        key: Subdirectory key (e.g., "by_id", "spatial0", "rel_segments")
        normalized_sharding: Normalized sharding configuration
        
    Returns:
        TensorStore sharded kvstore
    """
    base_path = output_path
    if not base_path.endswith("/") and not base_path.endswith(os.sep):
        base_path = base_path + "/"
    base_path = base_path + key + "/"
    
    if base_path.startswith("gs://"):
        parts = base_path[5:].split("/", 1)
        bucket = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        base_spec = {"driver": "gcs", "bucket": bucket, "path": path}
    elif base_path.startswith("s3://"):
        parts = base_path[5:].split("/", 1)
        bucket = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        base_spec = {"driver": "s3", "bucket": bucket, "path": path}
    else:
        abs_path = os.path.abspath(base_path)
        base_spec = {"driver": "file", "path": abs_path}
    
    kvstore_spec = {
        "driver": "neuroglancer_uint64_sharded",
        "base": base_spec,
        "metadata": normalized_sharding,
    }
    return ts.KvStore.open(kvstore_spec).result()


def _create_encoder_from_config_standalone(encoder_config: dict[str, Any]) -> AnnotationEncoder:
    """Create an AnnotationEncoder from config (standalone, can be pickled).
    
    Args:
        encoder_config: Encoder configuration dict with annotation_type, rank, properties
        
    Returns:
        AnnotationEncoder instance
    """
    from neuroglancer import viewer_state
    
    property_specs = [
        viewer_state.AnnotationPropertySpec(prop_dict)
        for prop_dict in encoder_config["properties"]
    ]
    return AnnotationEncoder(
        annotation_type=encoder_config["annotation_type"],
        rank=encoder_config["rank"],
        properties=property_specs,
    )


def _create_encoder_config_from_encoder(encoder: AnnotationEncoder) -> dict[str, Any]:
    """Create encoder config dict from an AnnotationEncoder instance.
    
    Args:
        encoder: AnnotationEncoder instance
        
    Returns:
        Encoder configuration dict
    """
    return {
        "annotation_type": encoder.annotation_type,
        "rank": encoder.rank,
        "properties": [
            {
                "id": prop.id,
                "type": prop.type,
                "description": getattr(prop, "description", None),
                "default": getattr(prop, "default", None),
                "enum_values": getattr(prop, "enum_values", None),
                "enum_labels": getattr(prop, "enum_labels", None),
            }
            for prop in encoder.properties
        ],
    }


def _sum_delayed_counts(written_counts_bag) -> int:
    """Sum counts from a Dask bag of [count] lists using delayed computation.
    
    Args:
        written_counts_bag: Dask bag where each partition returns [count]
        
    Returns:
        Total count
    """
    try:
        import dask
    except ImportError:
        raise ImportError(
            "Dask is required for distributed processing. "
            "Install with: pip install dask"
        )
    
    delayed_results = written_counts_bag.to_delayed()
    delayed_counts = [dask.delayed(lambda x: x[0] if x else 0)(d) for d in delayed_results]
    return dask.delayed(sum)(delayed_counts).compute()


class _WriteSpatialCellCallable:
    """Callable class to write a single spatial cell group that can be pickled for Dask.
    
    This processes individual cells, opening kvstores once (cached) and writing
    each cell. For sharded format, cells are batched by shard using transactions.
    """
    
    def __init__(
        self,
        output_path: str,
        level_num: int,
        grid_shape: tuple[int, ...],
        sharding_config: dict[str, Any] | None,
        encoder_config: dict[str, Any],
    ):
        self.output_path = output_path
        self.level_num = level_num
        self.grid_shape = grid_shape
        self.sharding_config = sharding_config
        self.encoder_config = encoder_config
    
    def __call__(self, cell_group: tuple[tuple[int, ...], list[EncodedAnnotation]]) -> int:
        """Write a single cell group.
        
        Args:
            cell_group: Tuple of (cell_coords, annotations)
            
        Returns:
            Number of annotations written
        """
        return _write_spatial_cell_standalone(
            cell_group=cell_group,
            output_path=self.output_path,
            level_num=self.level_num,
            grid_shape=self.grid_shape,
            sharding_config=self.sharding_config,
            encoder_config=self.encoder_config,
        )


def _get_cached_kvstore(cache_key: str, factory_func):
    """Get or create a cached kvstore on the worker."""
    if cache_key not in _worker_kvstore_cache:
        _worker_kvstore_cache[cache_key] = factory_func()
    return _worker_kvstore_cache[cache_key]


class _WriteShardGroupCallable:
    """Callable class to write shard groups that can be pickled for Dask."""
    def __init__(
        self,
        output_path: str,
        sharding_config: dict[str, Any] | None,
        encoder_config: dict[str, Any],
    ):
        self.output_path = output_path
        self.sharding_config = sharding_config
        self.encoder_config = encoder_config
    
    def __call__(self, group: tuple) -> int:
        """Write a shard group.
        
        Accepts either:
        - (shard_num, [annotations]) format (already extracted)
        - (shard_num, [(shard_num, ann), ...]) format (from groupby, needs extraction)
        
        Passes the group directly to _write_shard_group_standalone which will
        extract annotations inline during the write loop to avoid intermediate list creation.
        """
        return _write_shard_group_standalone(
            shard_group=group,
            output_path=self.output_path,
            sharding_config=self.sharding_config,
            encoder_config=self.encoder_config,
        )


def _write_shard_group_standalone(
    shard_group: tuple[int, list[EncodedAnnotation] | list[tuple[int, EncodedAnnotation]]],
    output_path: str,
    sharding_config: dict[str, Any] | None,
    encoder_config: dict[str, Any],
) -> int:
    """Standalone function to write a shard group that can be pickled for Dask.
    
    Creates TensorStore objects on the worker and writes annotations for a single shard.
    Extracts annotations inline during the write loop to avoid intermediate list creation.
    
    Args:
        shard_group: Tuple of (shard_num, items) where items can be:
            - List of EncodedAnnotation (already extracted)
            - List of (shard_num, EncodedAnnotation) tuples (from groupby, needs extraction)
        output_path: Path to output (gs://, s3://, or local)
        sharding_config: Sharding configuration (may be None for unsharded)
        encoder_config: Encoder configuration (annotation_type, rank, properties)
        
    Returns:
        Number of annotations written
    """
    import os
    import struct
    import time
    import tensorstore as ts
    from .encoder import AnnotationEncoder
    from neuroglancer import viewer_state
    
    shard_num, items = shard_group
    
    if not items:
        return 0
    
    # Check if items are (shard_num, ann) tuples from groupby or already annotations
    # We'll extract inline during the write loop to avoid creating an intermediate list
    needs_extraction = items and isinstance(items[0], tuple) and len(items[0]) == 2
    
    # Profile: Track time for encoder creation, kvstore creation, and writing
    setup_start = time.time()
    
    # Create encoder on worker from config
    property_specs = [
        viewer_state.AnnotationPropertySpec(prop_dict)
        for prop_dict in encoder_config["properties"]
    ]
    encoder = AnnotationEncoder(
        annotation_type=encoder_config["annotation_type"],
        rank=encoder_config["rank"],
        properties=property_specs,
    )
    
    # Get or create base kvstore (cached per worker)
    def create_base_kvstore():
        if output_path.startswith("gs://") or output_path.startswith("s3://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        elif output_path.startswith("file://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        else:
            abs_path = os.path.abspath(output_path)
            if not abs_path.endswith(os.sep):
                abs_path = abs_path + os.sep
            return ts.KvStore.open({"driver": "file", "path": abs_path}).result()
    
    base_cache_key = f"base_{output_path}"
    base_kvstore = _get_cached_kvstore(base_cache_key, create_base_kvstore)
    
    if sharding_config and "by_id" in sharding_config:
        # Sharded: use sharded kvstore
        by_id_sharding = sharding_config["by_id"]
        
        # Normalize sharding config
        normalized = {
            "@type": "neuroglancer_uint64_sharded_v1",
            "preshift_bits": by_id_sharding.get("preshift_bits", 0),
            "hash": by_id_sharding.get("hash", "murmurhash3_x86_128"),
            "minishard_bits": by_id_sharding.get("minishard_bits", 0),
            "shard_bits": by_id_sharding.get("shard_bits", 0),
            "minishard_index_encoding": by_id_sharding.get("minishard_index_encoding", "gzip"),
            "data_encoding": by_id_sharding.get("data_encoding", "gzip"),
        }
        
        # Convert num_shards to shard_bits if provided
        if "num_shards" in by_id_sharding:
            num_shards = by_id_sharding["num_shards"]
            shard_bits = 0
            while (1 << shard_bits) < num_shards:
                shard_bits += 1
            if (1 << shard_bits) != num_shards:
                raise ValueError(f"num_shards must be a power of 2, got {num_shards}")
            normalized["shard_bits"] = shard_bits
        
        # Get or create sharded kvstore (cached per worker)
        # Use the same format as _get_sharded_kvstore method
        def create_by_id_kvstore():
            # Construct base spec with by_id subdirectory
            # Use the same approach as _get_sharded_kvstore
            import os
            base_path = output_path
            if not base_path.endswith("/") and not base_path.endswith(os.sep):
                base_path = base_path + "/"
            base_path = base_path + "by_id/"
            
            # Create base kvstore spec (same logic as _get_sharded_kvstore)
            if base_path.startswith("gs://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "gcs", "bucket": bucket, "path": path}
            elif base_path.startswith("s3://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "s3", "bucket": bucket, "path": path}
            else:
                abs_path = os.path.abspath(base_path)
                base_spec = {"driver": "file", "path": abs_path}
            
            kvstore_spec = {
                "driver": "neuroglancer_uint64_sharded",
                "base": base_spec,
                "metadata": normalized,
            }
            return ts.KvStore.open(kvstore_spec).result()
        
        by_id_cache_key = f"by_id_{output_path}"
        by_id_kvstore = _get_cached_kvstore(by_id_cache_key, create_by_id_kvstore)
        setup_time = time.time() - setup_start
        
        # Write all annotations for this shard in a transaction
        # Extract annotations inline during write loop to avoid intermediate list
        write_start = time.time()
        count = 0
        encode_time = 0
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = by_id_kvstore.with_transaction(txn)
            if needs_extraction:
                # Extract annotation from (shard_num, ann) tuple inline
                for _, ann in items:
                    encode_start = time.time()
                    encoded = encoder.encode_single_with_relationships(ann)
                    encode_time += time.time() - encode_start
                    key_bytes = struct.pack(">Q", ann.id)
                    kvstore_txn[key_bytes] = encoded
                    count += 1
            else:
                # Already annotations, write directly
                for ann in items:
                    encode_start = time.time()
                    encoded = encoder.encode_single_with_relationships(ann)
                    encode_time += time.time() - encode_start
                    key_bytes = struct.pack(">Q", ann.id)
                    kvstore_txn[key_bytes] = encoded
                    count += 1
        write_time = time.time() - write_start
        
        # Log profiling info (only for larger shard groups to avoid spam)
        if count > 100:
            logger.info(
                f"by_id: Shard {shard_num}: {count} annotations, "
                f"setup={setup_time*1000:.1f}ms, encode={encode_time*1000:.1f}ms, "
                f"write={write_time*1000:.1f}ms"
            )
        else:
            logger.info(f"by_id: Shard {shard_num}: {count} annotations")
        
    else:
        # Unsharded: use base kvstore with string keys
        # Extract annotations inline during write loop to avoid intermediate list
        setup_time = time.time() - setup_start
        write_start = time.time()
        count = 0
        encode_time = 0
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = base_kvstore.with_transaction(txn)
            if needs_extraction:
                # Extract annotation from (shard_num, ann) tuple inline
                for _, ann in items:
                    encode_start = time.time()
                    encoded = encoder.encode_single_with_relationships(ann)
                    encode_time += time.time() - encode_start
                    key = f"by_id/{ann.id}"
                    key_bytes = key.encode("utf-8")
                    kvstore_txn[key_bytes] = encoded
                    count += 1
            else:
                # Already annotations, write directly
                for ann in items:
                    encode_start = time.time()
                    encoded = encoder.encode_single_with_relationships(ann)
                    encode_time += time.time() - encode_start
                    key = f"by_id/{ann.id}"
                    key_bytes = key.encode("utf-8")
                    kvstore_txn[key_bytes] = encoded
                    count += 1
        write_time = time.time() - write_start
        
        # Log profiling info (only for larger shard groups to avoid spam)
        if count > 100:
            logger.info(
                f"by_id: Unsharded shard {shard_num}: {count} annotations, "
                f"setup={setup_time*1000:.1f}ms, encode={encode_time*1000:.1f}ms, "
                f"write={write_time*1000:.1f}ms"
            )
        else:
            logger.info(f"by_id: Unsharded shard {shard_num}: {count} annotations")
    
    return count


def _write_spatial_cell_standalone(
    cell_group: tuple[tuple[int, ...], list[EncodedAnnotation]],
    output_path: str,
    level_num: int,
    grid_shape: tuple[int, ...],
    sharding_config: dict[str, Any] | None,
    encoder_config: dict[str, Any],
) -> int:
    """Standalone function to write a single spatial cell group.
    
    This function:
    1. Opens kvstores once (cached per worker)
    2. Encodes and writes a single cell
    3. Reuses encoder and kvstore objects
    
    Args:
        cell_group: Tuple of (cell_coords, annotations)
        output_path: Path to output (gs://, s3://, or local)
        level_num: Spatial level number
        grid_shape: Grid shape for this level
        sharding_config: Sharding configuration (may be None)
        encoder_config: Encoder configuration (annotation_type, rank, properties)
        
    Returns:
        Number of annotations written
    """
    import os
    import struct
    import tensorstore as ts
    from .encoder import AnnotationEncoder
    from neuroglancer import viewer_state
    
    cell_coords, annotations = cell_group
    
    if not annotations:
        return 0
    
    # Create encoder (cached per worker via module-level cache if needed)
    property_specs = [
        viewer_state.AnnotationPropertySpec(prop_dict)
        for prop_dict in encoder_config["properties"]
    ]
    encoder = AnnotationEncoder(
        annotation_type=encoder_config["annotation_type"],
        rank=encoder_config["rank"],
        properties=property_specs,
    )
    
    # Encode annotations
    encoded = encoder.encode_multiple(annotations)
    
    # Get or create base kvstore (cached per worker)
    def create_base_kvstore():
        if output_path.startswith("gs://") or output_path.startswith("s3://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        elif output_path.startswith("file://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        else:
            abs_path = os.path.abspath(output_path)
            if not abs_path.endswith(os.sep):
                abs_path = abs_path + os.sep
            return ts.KvStore.open({"driver": "file", "path": abs_path}).result()
    
    base_cache_key = f"base_{output_path}"
    base_kvstore = _get_cached_kvstore(base_cache_key, create_base_kvstore)
    
    spatial_key = f"spatial{level_num}"
    spatial_sharding = None
    if sharding_config and "spatial" in sharding_config:
        spatial_sharding = sharding_config["spatial"]
    
    if spatial_sharding:
        # Sharded: use compressed morton code as uint64 key
        
        # Normalize sharding config
        normalized = {
            "@type": "neuroglancer_uint64_sharded_v1",
            "preshift_bits": spatial_sharding.get("preshift_bits", 0),
            "hash": spatial_sharding.get("hash", "murmurhash3_x86_128"),
            "minishard_bits": spatial_sharding.get("minishard_bits", 0),
            "shard_bits": spatial_sharding.get("shard_bits", None),
            "minishard_index_encoding": spatial_sharding.get("minishard_index_encoding", "gzip"),
            "data_encoding": spatial_sharding.get("data_encoding", "gzip"),
        }
        
        # Convert num_shards to shard_bits if provided
        if "num_shards" in spatial_sharding:
            if normalized["shard_bits"] is not None:
                raise ValueError("Cannot specify both 'num_shards' and 'shard_bits' in spatial sharding config")
            num_shards = spatial_sharding["num_shards"]
            shard_bits = 0
            while (1 << shard_bits) < num_shards:
                shard_bits += 1
            if (1 << shard_bits) != num_shards:
                raise ValueError(f"num_shards must be a power of 2, got {num_shards}")
            normalized["shard_bits"] = shard_bits
        
        # Ensure shard_bits is set
        if normalized["shard_bits"] is None:
            normalized["shard_bits"] = 0  # Default to 1 shard (unsharded) if not specified
        
        # Get or create sharded kvstore (cached per worker)
        # Use the same format as _get_sharded_kvstore method
        def create_spatial_kvstore():
            # Construct base spec with spatial subdirectory
            # Use the same approach as _get_sharded_kvstore
            import os
            base_path = output_path
            if not base_path.endswith("/") and not base_path.endswith(os.sep):
                base_path = base_path + "/"
            base_path = base_path + spatial_key + "/"
            
            # Create base kvstore spec (same logic as _get_sharded_kvstore)
            if base_path.startswith("gs://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "gcs", "bucket": bucket, "path": path}
            elif base_path.startswith("s3://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "s3", "bucket": bucket, "path": path}
            else:
                abs_path = os.path.abspath(base_path)
                base_spec = {"driver": "file", "path": abs_path}
            
            kvstore_spec = {
                "driver": "neuroglancer_uint64_sharded",
                "base": base_spec,
                "metadata": normalized,
            }
            return ts.KvStore.open(kvstore_spec).result()
        
        spatial_cache_key = f"{spatial_key}_{output_path}"
        spatial_kvstore = _get_cached_kvstore(spatial_cache_key, create_spatial_kvstore)
        
        morton_code = _zorder_compressed(cell_coords, grid_shape)
        key_bytes = struct.pack(">Q", morton_code)
        
        # Use transaction for atomic write
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = spatial_kvstore.with_transaction(txn)
            kvstore_txn[key_bytes] = encoded
            # Transaction commits automatically when exiting context
    else:
        # Unsharded: use string key
        cell_key_parts = [str(c) for c in cell_coords]
        cell_key = "_".join(cell_key_parts)
        key = f"{spatial_key}/{cell_key}"
        key_bytes = key.encode("utf-8")
        
        # Use transaction for atomic write (even though it's a single file)
        # This ensures the write is atomic and provides consistency guarantees
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = base_kvstore.with_transaction(txn)
            kvstore_txn[key_bytes] = encoded
            # Transaction commits automatically when exiting context
    
    return len(annotations)


def _write_relationship_group_standalone(
    rel_group: tuple[tuple[str, int], list[EncodedAnnotation]],
    output_path: str,
    sharding_config: dict[str, Any] | None,
    encoder_config: dict[str, Any],
) -> int:
    """Standalone function to write a relationship group that can be pickled for Dask.
    
    Args:
        rel_group: Tuple of ((relationship, object_id), annotations)
        output_path: Path to output (gs://, s3://, or local)
        sharding_config: Sharding configuration (may be None)
        encoder_config: Encoder configuration (annotation_type, rank, properties)
        
    Returns:
        Number of annotations written
    """
    import os
    import struct
    import tensorstore as ts
    from .encoder import AnnotationEncoder
    from neuroglancer import viewer_state
    
    (relationship, object_id), annotations = rel_group
    
    if not annotations:
        return 0
    
    # Create encoder on worker from config
    property_specs = [
        viewer_state.AnnotationPropertySpec(prop_dict)
        for prop_dict in encoder_config["properties"]
    ]
    encoder = AnnotationEncoder(
        annotation_type=encoder_config["annotation_type"],
        rank=encoder_config["rank"],
        properties=property_specs,
    )
    
    # Encode annotations
    encoded = encoder.encode_multiple(annotations)
    
    # Get or create base kvstore (cached per worker)
    def create_base_kvstore():
        if output_path.startswith("gs://") or output_path.startswith("s3://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        elif output_path.startswith("file://"):
            path = output_path if output_path.endswith("/") else output_path + "/"
            return ts.KvStore.open(path).result()
        else:
            abs_path = os.path.abspath(output_path)
            if not abs_path.endswith(os.sep):
                abs_path = abs_path + os.sep
            return ts.KvStore.open({"driver": "file", "path": abs_path}).result()
    
    base_cache_key = f"base_{output_path}"
    base_kvstore = _get_cached_kvstore(base_cache_key, create_base_kvstore)
    
    rel_config = sharding_config.get("relationships", {}).get(relationship) if sharding_config else None
    
    if rel_config:
        # Sharded: use uint64 key
        rel_key = f"rel_{relationship}"
        
        # Normalize sharding config (inline normalization logic to avoid creating writer instance)
        normalized = {
            "@type": "neuroglancer_uint64_sharded_v1",
            "preshift_bits": rel_config.get("preshift_bits", 0),
            "hash": rel_config.get("hash", "murmurhash3_x86_128"),
            "minishard_bits": rel_config.get("minishard_bits", 0),
            "shard_bits": rel_config.get("shard_bits", None),
            "minishard_index_encoding": rel_config.get("minishard_index_encoding", "gzip"),
            "data_encoding": rel_config.get("data_encoding", "gzip"),
        }
        
        # Convert num_shards to shard_bits if provided
        if "num_shards" in rel_config:
            if normalized["shard_bits"] is not None:
                raise ValueError("Cannot specify both 'num_shards' and 'shard_bits'")
            num_shards = rel_config["num_shards"]
            shard_bits = 0
            while (1 << shard_bits) < num_shards:
                shard_bits += 1
            if (1 << shard_bits) != num_shards:
                raise ValueError(f"num_shards must be a power of 2, got {num_shards}")
            normalized["shard_bits"] = shard_bits
        
        # Ensure shard_bits is set
        if normalized["shard_bits"] is None:
            raise ValueError("Must specify either 'num_shards' or 'shard_bits' in relationship sharding config")
        
        # Get or create sharded kvstore (cached per worker)
        def create_rel_kvstore():
            import os
            base_path = output_path
            if not base_path.endswith("/") and not base_path.endswith(os.sep):
                base_path = base_path + "/"
            base_path = base_path + rel_key + "/"
            
            if base_path.startswith("gs://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "gcs", "bucket": bucket, "path": path}
            elif base_path.startswith("s3://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "s3", "bucket": bucket, "path": path}
            else:
                abs_path = os.path.abspath(base_path)
                base_spec = {"driver": "file", "path": abs_path}
            
            kvstore_spec = {
                "driver": "neuroglancer_uint64_sharded",
                "base": base_spec,
                "metadata": normalized,
            }
            return ts.KvStore.open(kvstore_spec).result()
        
        rel_cache_key = f"{rel_key}_{output_path}"
        rel_kvstore = _get_cached_kvstore(rel_cache_key, create_rel_kvstore)
        
        key_bytes = struct.pack(">Q", object_id)
        
        # Use transaction for atomic write
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = rel_kvstore.with_transaction(txn)
            kvstore_txn[key_bytes] = encoded
    else:
        # Unsharded: use string key
        rel_key = f"rel_{relationship}"
        key = f"{rel_key}/{object_id}"
        key_bytes = key.encode("utf-8")
        
        # Use transaction for atomic write
        with ts.Transaction(atomic=True) as txn:
            kvstore_txn = base_kvstore.with_transaction(txn)
            kvstore_txn[key_bytes] = encoded
    
    return len(annotations)


class _WriteRelationshipGroupCallable:
    """Callable class to write relationship groups that can be pickled for Dask."""
    def __init__(
        self,
        output_path: str,
        sharding_config: dict[str, Any] | None,
        encoder_config: dict[str, Any],
    ):
        self.output_path = output_path
        self.sharding_config = sharding_config
        self.encoder_config = encoder_config
    
    def __call__(self, rel_group: tuple[tuple[str, int], list[EncodedAnnotation]]) -> int:
        """Write a relationship group.
        
        Args:
            rel_group: Tuple of ((relationship, object_id), annotations)
            
        Returns:
            Number of annotations written
        """
        return _write_relationship_group_standalone(
            rel_group=rel_group,
            output_path=self.output_path,
            sharding_config=self.sharding_config,
            encoder_config=self.encoder_config,
        )


class DistributedAnnotationWriter:
    """Writer for precomputed annotations using TensorStore.
    
    This writer supports both sharded and unsharded formats, and can write
    directly to cloud storage (GCS, S3) or local filesystems.
    """
    
    def __init__(
        self,
        output_path: str,
        sharding_config: dict[str, Any] | None = None,
        encoder_config: dict[str, Any] | None = None,
    ):
        """Initialize the writer.
        
        Args:
            output_path: Base path for output (e.g., "gs://bucket/path" or "/tmp/output")
            sharding_config: Configuration for sharding (e.g., {"by_id": {"num_shards": 16}})
            encoder_config: Configuration for encoder (annotation_type, rank, properties)
        """
        self.output_path = output_path
        self.sharding_config = sharding_config if sharding_config else {}
        
        # Initialize encoder if config provided
        if encoder_config:
            from neuroglancer import viewer_state
            property_specs = [
                viewer_state.AnnotationPropertySpec(prop_dict)
                for prop_dict in encoder_config["properties"]
            ]
            self.encoder = AnnotationEncoder(
                annotation_type=encoder_config["annotation_type"],
                rank=encoder_config["rank"],
                properties=property_specs,
            )
        else:
            self.encoder = None
        
        # Store encoder_config for distributed functions
        self.encoder_config = encoder_config
        
        # Initialize kvstores lazily (created on first use)
        self.base_kvstore: ts.KvStore | None = None
        self.by_id_kvstore: ts.KvStore | None = None
        
        # Store spatial_limit (default 1000, can be overridden by spatial_index_config)
        # This is used by _get_spatial_limit() method
        self.spatial_limit = 1000
    
    def _open_kvstore(self, path: str) -> ts.KvStore:
        """Open TensorStore kvstore for the given path."""
        import os
        
        if path.startswith("gs://") or path.startswith("s3://"):
            if not path.endswith("/"):
                path = path + "/"
            return ts.KvStore.open(path).result()
        elif path.startswith("file://"):
            if not path.endswith("/"):
                path = path + "/"
            return ts.KvStore.open(path).result()
        else:
            abs_path = os.path.abspath(path)
            if not abs_path.endswith(os.sep):
                abs_path = abs_path + os.sep
            return ts.KvStore.open({"driver": "file", "path": abs_path}).result()
    
    def _get_sharded_kvstore(self, index_name: str, key: str) -> ts.KvStore:
        """Get a sharded kvstore for an index if sharding is enabled, otherwise return base kvstore."""
        sharding = None
        if index_name in self.sharding_config:
            sharding = self.sharding_config[index_name]
        elif index_name == "spatial" and "spatial" in self.sharding_config:
            sharding = self.sharding_config["spatial"]
        elif index_name.startswith("rel_") and "relationships" in self.sharding_config:
            rel_name = index_name[4:]
            if rel_name in self.sharding_config.get("relationships", {}):
                sharding = self.sharding_config["relationships"][rel_name]

        if sharding:
            normalized_sharding = self._normalize_sharding_config(sharding)
            import os
            
            base_path = self.output_path
            if not base_path.endswith("/") and not base_path.endswith(os.sep):
                base_path = base_path + "/"
            base_path = base_path + key + "/"
            
            if base_path.startswith("gs://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "gcs", "bucket": bucket, "path": path}
            elif base_path.startswith("s3://"):
                parts = base_path[5:].split("/", 1)
                bucket = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                base_spec = {"driver": "s3", "bucket": bucket, "path": path}
            else:
                abs_path = os.path.abspath(base_path)
                base_spec = {"driver": "file", "path": abs_path}
            
            return ts.KvStore.open(
                {
                    "driver": "neuroglancer_uint64_sharded",
                    "base": base_spec,
                    "metadata": normalized_sharding,
                }
            ).result()
        else:
            return self.base_kvstore
    
    def _get_spatial_limit(self) -> int:
        """Get spatial index limit from config."""
        return self.spatial_limit
    
    def _normalize_sharding_config(self, sharding: dict[str, Any]) -> dict[str, Any]:
        """Normalize sharding config (delegates to standalone function)."""
        return _normalize_sharding_config_standalone(sharding)
    
    def _build_spatial_level_config(
        self,
        level: SpatialIndexLevel,
        level_sharding_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build spatial level configuration dictionary.
        
        Args:
            level: Spatial index level
            level_sharding_config: Level-specific sharding config (for adaptive sharding), or None to use base config
            
        Returns:
            Dictionary with level configuration including sharding if applicable
        """
        config = {
            "key": f"spatial{level.level}",
            "grid_shape": list(level.grid_shape),
            "chunk_size": list(level.chunk_size),
            "limit": self._get_spatial_limit(),
        }
        
        # Determine which sharding config to use
        if level_sharding_config is not None:
            # Use level-specific sharding config (for adaptive sharding)
            sharding = level_sharding_config
        else:
            # Fall back to base config, but only if it doesn't use adaptive sharding
            base_spatial = self.sharding_config.get("spatial") if self.sharding_config else None
            if base_spatial and "max_annotations_per_shard" not in base_spatial:
                # Base config doesn't use adaptive sharding, safe to use
                sharding = base_spatial
            else:
                # Base config uses adaptive sharding but no level-specific config available
                # This shouldn't happen if adaptive sharding is working correctly, but skip sharding
                sharding = None
        
        if sharding:
            config["sharding"] = self._normalize_sharding_config(sharding)
        
        return config

    def write_info(
        self,
        coordinate_space: Any,
        annotation_type: str,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        properties: Sequence[Any],
        relationships: Sequence[str],
        spatial_levels: Sequence[SpatialIndexLevel],
        level_sharding_configs: dict[int, dict[str, Any]] | None = None,
    ):
        """Write the info JSON file.

        Args:
            coordinate_space: Coordinate space specification
            annotation_type: Type of annotation
            lower_bound: Lower bound of coordinate space
            upper_bound: Upper bound of coordinate space
            properties: Property specifications
            relationships: Relationship names
            spatial_levels: Spatial index levels
        """
        # Convert annotation type
        type_map = {
            "point": "POINT",
            "line": "LINE",
            "polyline": "POLYLINE",
            "axis_aligned_bounding_box": "AXIS_ALIGNED_BOUNDING_BOX",
            "ellipsoid": "ELLIPSOID",
        }
        annotation_type_upper = type_map.get(annotation_type, annotation_type.upper())

        info = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": coordinate_space.to_json() if hasattr(coordinate_space, "to_json") else coordinate_space,
            "lower_bound": list(lower_bound),
            "upper_bound": list(upper_bound),
            "annotation_type": annotation_type_upper,
            "properties": [p.to_json() if hasattr(p, "to_json") else p for p in properties],
            "relationships": [
                {
                    "id": rel,
                    "key": f"rel_{rel}",
                    **(
                        {"sharding": self._normalize_sharding_config(self.sharding_config.get("relationships", {}).get(rel))}
                        if self.sharding_config.get("relationships", {}).get(rel)
                        else {}
                    ),
                }
                for rel in relationships
            ],
            "by_id": {
                "key": "by_id",
                **(
                    {"sharding": self._normalize_sharding_config(self.sharding_config.get("by_id"))}
                    if self.sharding_config.get("by_id")
                    else {}
                ),
            },
            "spatial": [
                self._build_spatial_level_config(
                    level,
                    level_sharding_configs.get(level.level) if level_sharding_configs else None,
                )
                for level in spatial_levels
            ],
        }

        # Initialize base kvstore if needed
        if self.base_kvstore is None:
            self.base_kvstore = self._open_kvstore(self.output_path)
        
        # Write info file
        info_json = json.dumps(info, indent=2)
        self.base_kvstore.write("info", info_json.encode("utf-8")).result()

    def compute_shard_number(self, annotation_id: int) -> int:
        """Compute shard number for an annotation ID.
        
        This method can be used in distributed processing (e.g., Dask map operations)
        to compute shard numbers without loading all annotations into memory.
        
        IMPORTANT: This uses the normalized sharding config to match TensorStore's calculation.
        
        Args:
            annotation_id: Annotation ID
            
        Returns:
            Shard number
        """
        if "by_id" not in self.sharding_config:
            return 0  # Unsharded
        
        # Use normalized sharding config to match what TensorStore uses
        raw_sharding = self.sharding_config["by_id"]
        sharding = self._normalize_sharding_config(raw_sharding)
        
        preshift_bits = sharding.get("preshift_bits", 0)
        hash_func = sharding.get("hash", "identity")
        minishard_bits = sharding.get("minishard_bits", 0)
        shard_bits = sharding.get("shard_bits", 0)
        
        chunk_id = annotation_id
        # Compute hashed chunk ID
        if hash_func == "identity":
            hashed_id = chunk_id >> preshift_bits
        elif hash_func == "murmurhash3_x86_128":
            # Use mmh3 for MurmurHash3_x86_128
            # According to spec: "applied to the shifted chunk ID in little endian encoding"
            # "The low 8 bytes of the resultant hash code are treated as a little endian 64-bit number"
            chunk_id_shifted = chunk_id >> preshift_bits
            # Convert to 8 bytes (64 bits) in little-endian format
            # Always use exactly 8 bytes, even if the value is small
            chunk_id_bytes = chunk_id_shifted.to_bytes(8, 'little', signed=False)
            # mmh3.hash128 returns a tuple (hash_low_64bits, hash_high_64bits) for 128-bit hash
            # According to spec: "The low 8 bytes of the resultant hash code are treated as a little endian 64-bit number"
            # So we use the first element (low 64 bits)
            hash_result = mmh3.hash128(chunk_id_bytes, x64arch=False)
            if isinstance(hash_result, tuple):
                # hash_result is (low_64bits, high_64bits)
                # We want the low 64 bits, but need to ensure it's treated as unsigned
                hashed_id = hash_result[0]
                # Convert signed to unsigned if needed (Python ints are signed)
                if hashed_id < 0:
                    hashed_id = hashed_id + (1 << 64)
                hashed_id = hashed_id & ((1 << 64) - 1)
            else:
                # Single value (shouldn't happen with hash128, but handle it)
                hashed_id = hash_result & ((1 << 64) - 1)
        else:
            raise ValueError(f"Unsupported hash function: {hash_func}")
        
        # Extract shard number
        # Shard number = bits [minishard_bits, minishard_bits + shard_bits) of hashed_id
        # This matches the TypeScript implementation in sharded.ts line 89-91:
        # const shard = ((1n << BigInt(sharding.shardBits)) - 1n) &
        #                (shardAndMinishard >> BigInt(sharding.minishardBits));
        # where shardAndMinishard = hashCode & ((1n << BigInt(minishardBits + shardBits)) - 1n)
        shard_and_minishard = hashed_id & ((1 << (minishard_bits + shard_bits)) - 1)
        shard = (shard_and_minishard >> minishard_bits) & ((1 << shard_bits) - 1)
        return shard

    def write_annotations_by_id_batch(
        self, annotations: Sequence[EncodedAnnotation]
    ):
        """Write multiple annotations to the by_id index, handling sharding if enabled.

        Groups annotations by shard and writes each shard in a batch transaction.
        For distributed processing, use write_annotations_by_id_distributed() instead.

        Args:
            annotations: List of encoded annotations (must fit in memory)
        """
        from collections import defaultdict

        # Group annotations by shard number for parallelization
        # This allows workers to process all annotations for a given shard together
        shard_groups: dict[int, list[EncodedAnnotation]] = defaultdict(list)

        for ann in annotations:
            shard = self.compute_shard_number(ann.id)
            shard_groups[shard].append(ann)

        # Write each shard group in a batch transaction
        self._write_shard_group_batch(shard_groups)

    def _write_shard_group_batch(self, shard_groups: dict[int, list[EncodedAnnotation]]):
        """Write a dictionary of shard groups to the by_id index.
        
        This method can be called with pre-grouped annotations (e.g., from Dask groupby).
        
        Uses one transaction per shard group. Since all annotations in a shard group
        map to the same shard file, this allows TensorStore to batch writes efficiently
        without file locking on each individual write.
        
        Args:
            shard_groups: Dictionary mapping shard number to list of annotations
        """
        import struct
        import tensorstore as ts
        
        # Initialize kvstores if needed
        if self.base_kvstore is None:
            self.base_kvstore = self._open_kvstore(self.output_path)
        if self.by_id_kvstore is None:
            self.by_id_kvstore = self._get_sharded_kvstore("by_id", "by_id")
        
        # Use one transaction per shard group
        # All annotations in a shard group map to the same shard file,
        # allowing TensorStore to batch writes efficiently
        for shard_num, anns in shard_groups.items():
            with ts.Transaction(atomic=True) as txn:
                kvstore_txn = self.by_id_kvstore.with_transaction(txn)
                
                # Collect all writes for this shard in the transaction
                for ann in anns:
                    encoded = self.encoder.encode_single_with_relationships(ann)
                    # For sharded format, encode uint64 key as big-endian bytes (same as reader)
                    # TensorStore's sharded driver expects bytes keys
                    key_bytes = struct.pack(">Q", ann.id)
                    # Write to transaction - TensorStore batches these writes to the same shard file
                    kvstore_txn[key_bytes] = encoded
                
                # Transaction commits automatically when exiting the context
                # All writes to this shard file are batched together

    def write_annotations_by_id_distributed(
        self, annotations: Any  # Dask bag or similar distributed collection
    ):
        """Write annotations using distributed processing (Dask).
        
        This method writes annotations directly from workers without collecting
        them to the main process, avoiding serialization/transfer overhead.
        
        The distributed processing flow:
        1. Map: Compute shard number for each annotation (distributed, parallel)
        2. GroupBy: Group annotations by shard (distributed, lazy, uses hash partitioning)
        3. Map: Write each shard group directly from workers (no collection to main process)
        
        The key efficiency: 
        - Shard computation happens in parallel across workers
        - Grouping happens in parallel (hash partitioning)
        - Writing happens directly from workers (TensorStore objects created on workers)
        - No data collection to main process
        
        Args:
            annotations: Dask bag or similar distributed collection of EncodedAnnotation
            
        Returns:
            Total number of annotations written
        """
        try:
            import dask.bag as db
        except ImportError:
            raise ImportError(
                "Dask is required for distributed processing. "
                "Install with: pip install dask"
            )
        
        # Convert to Dask bag if not already
        # Use larger partition size (50K) to reduce task overhead and improve worker utilization
        # Smaller partitions create too many small tasks, increasing scheduler overhead
        if not isinstance(annotations, db.Bag):
            annotations = db.from_sequence(annotations, partition_size=50000)
        
        # Prepare encoder config for serialization
        encoder_config = {
            "annotation_type": self.encoder.annotation_type,
            "rank": self.encoder.rank,
            "properties": [
                {
                    "id": prop.id,
                    "type": prop.type,
                    "description": getattr(prop, "description", None),
                    "default": getattr(prop, "default", None),
                    "enum_values": getattr(prop, "enum_values", None),
                    "enum_labels": getattr(prop, "enum_labels", None),
                }
                for prop in self.encoder.properties
            ],
        }
        
        # Handle unsharded case
        if "by_id" not in self.sharding_config:
            logger.info("by_id: Unsharded mode - writing individual annotation files with async writes...")
            step1_start = time.time()
            
            # For unsharded by_id, each annotation is written to its own file (by_id/<id>)
            # Use async writes instead of transactions for better performance with many files
            output_path_param = self.output_path  # Capture for closure
            
            def process_unsharded_partition(partition):
                """Process a partition: write each annotation to its own file using async writes.
                
                This function runs on each worker and processes an entire partition.
                Each annotation is written to by_id/<id> as a separate file using async writes.
                
                Returns:
                    Total number of annotations written from this partition
                """
                # Create encoder on worker
                encoder = _create_encoder_from_config_standalone(encoder_config)
                
                # Get or create base kvstore (cached per worker)
                base_cache_key = f"base_{output_path_param}"
                base_kvstore = _get_cached_kvstore(
                    base_cache_key,
                    lambda: _create_base_kvstore_standalone(output_path_param)
                )
                
                # Collect futures for all writes
                futures = []
                for ann in partition:
                    encoded = encoder.encode_single_with_relationships(ann)
                    key = f"by_id/{ann.id}"
                    key_bytes = key.encode("utf-8")
                    # Use async write (returns a future)
                    future = base_kvstore.write(key_bytes, encoded)
                    futures.append(future)
                
                # Wait for all writes to complete
                for future in futures:
                    future.result()
                
                return len(futures)
            
            # Process each partition independently
            def process_partition_wrapper(partition):
                """Wrapper that returns an iterable (list) from the partition processing function."""
                count = process_unsharded_partition(partition)
                return [count]
            
            written_counts_bag = annotations.map_partitions(process_partition_wrapper)
            total_written = _sum_delayed_counts(written_counts_bag)
            
            step1_time = time.time() - step1_start
            logger.info(f"by_id: Processing and writing took {step1_time:.2f}s")
            return total_written
        
        # Get normalized sharding config for standalone computation
        raw_sharding = self.sharding_config["by_id"]
        sharding_config = self._normalize_sharding_config(raw_sharding)
        
        # Normalize sharding config once (before sending to workers)
        normalized_sharding_config = self._normalize_sharding_config(sharding_config)
        output_path_param = self.output_path  # Capture for closure
        
        # OPTIMIZATION: Shuffle by shard before writing to eliminate transaction contention.
        # Step 1: Group annotations by shard locally within each partition
        logger.info("by_id: Step 1 - Grouping annotations by shard locally within partitions...")
        step1_start = time.time()
        
        def group_by_shard_locally(partition):
            """Group annotations by shard within a partition - runs on worker.
            
            Returns:
                List of (shard, [annotations]) tuples, wrapped in a list
            """
            from collections import defaultdict
            shard_groups = defaultdict(list)
            for ann in partition:
                shard = _compute_shard_number_standalone(ann, normalized_sharding_config)
                shard_groups[shard].append(ann)
            # Return list of (shard, [anns]) tuples, wrapped in list (like relationship index pattern)
            return [[(shard, anns) for shard, anns in shard_groups.items() if anns]]
        
        # Map partitions to get locally grouped shard groups
        # Each partition returns [[(shard, [anns]), ...]] - wrapped in list to prevent Dask from iterating
        shard_groups_partitions = annotations.map_partitions(group_by_shard_locally)
        
        step1_time = time.time() - step1_start
        logger.info(f"by_id: Step 1 (local grouping) took {step1_time:.2f}s")
        
        # Step 2: Extract groups from wrapped format, convert to bag of (shard, [anns]) tuples
        # Use tuple key (shard,) instead of int to match relationship index pattern and avoid flatten() issues
        logger.info("by_id: Step 2 - Extracting groups and shuffling globally...")
        step2_start = time.time()
        
        def extract_shard_groups_from_partition(wrapped_result):
            """Extract list of (shard, [anns]) tuples from wrapped partition result.
            
            Converts shard to tuple key (shard,) to match relationship index pattern.
            """
            # wrapped_result is [[(shard, [anns]), ...]] - unwrap one level
            if isinstance(wrapped_result, list) and wrapped_result:
                inner_list = wrapped_result[0] if isinstance(wrapped_result[0], list) else wrapped_result
                # Convert (shard, [anns]) to ((shard,), [anns]) to use tuple key (like relationship index)
                return [((shard,), anns) for shard, anns in inner_list]
            return []
        
        # Extract groups and convert to bag - use tuple key (shard,) instead of int
        shard_groups_bag = shard_groups_partitions.map(extract_shard_groups_from_partition).flatten()
        
        # Group by shard globally - now shuffling groups instead of individual annotations!
        # OPTIMIZATION: Shuffling ~32-128 groups per partition instead of 5M annotations (~100x reduction)
        annotations_grouped = shard_groups_bag.groupby(lambda x: x[0])  # Group by (shard,) tuple key
        
        # Merge annotation lists from groupby result
        def merge_shard_groups(shard_group):
            """Merge annotation lists from groupby result - runs on worker."""
            (shard,), items = shard_group  # items is list of ((shard,), [anns]) tuples
            # Merge all annotation lists into one
            all_annotations = []
            for _, anns_list in items:
                all_annotations.extend(anns_list)
            return ((shard,), all_annotations)  # Return with tuple key for consistency
        
        # Merge groups - result is bag of ((shard,), [all_anns_for_shard]) tuples
        shard_groups_merged = annotations_grouped.map(merge_shard_groups)
        
        step2_time = time.time() - step2_start
        logger.info(f"by_id: Step 2 (global grouping and merging) took {step2_time:.2f}s")
        
        # Step 3: Extract annotations from merged groups and flatten to individual annotations
        # shard_groups_merged is a bag of ((shard,), [anns]) tuples - annotations are already in lists
        # Extract the annotation lists and flatten to get a bag of individual annotations
        logger.info("by_id: Step 3 - Extracting annotations from merged groups...")
        step3_start = time.time()
        
        def extract_annotations_from_merged_group(merged_group_item):
            """Extract annotation list from merged group item - runs on worker."""
            (shard,), anns = merged_group_item  # anns is already a list of annotations
            return anns  # Return the list directly
        
        # Extract annotation lists and flatten to individual annotations
        # After this, annotations are partitioned by shard (all annotations for a shard are together)
        annotations = shard_groups_merged.map(extract_annotations_from_merged_group).flatten()
        
        step3_time = time.time() - step3_start
        logger.info(f"by_id: Step 3 (extract and flatten) took {step3_time:.2f}s")
        
        # OPTIMIZED APPROACH: Process partitions locally, write each shard in a transaction.
        # After shuffling by shard, each partition only contains annotations for specific shards,
        # eliminating transaction contention.
        logger.info("by_id: Processing partitions with transactional writes...")
        step1_start = time.time()
        
        def process_partition_with_transactions(partition):
            """Process a partition: write each shard group in a transaction.
            
            This function runs on each worker and processes an entire partition.
            After shuffling by shard, each partition only contains annotations for
            specific shards, so we can write each shard in a single transaction without contention.
            
            Returns:
                Total number of annotations written from this partition
            """
            # Create encoder on worker
            encoder = _create_encoder_from_config_standalone(encoder_config)
            
            # Get or create base kvstore (cached per worker)
            base_cache_key = f"base_{output_path_param}"
            base_kvstore = _get_cached_kvstore(
                base_cache_key,
                lambda: _create_base_kvstore_standalone(output_path_param)
            )
            
            # Get or create sharded kvstore for by_id (cached per worker)
            # TensorStore will automatically route writes to the correct shard files
            by_id_cache_key = f"by_id_{output_path_param}"
            by_id_kvstore = _get_cached_kvstore(
                by_id_cache_key,
                lambda: _create_sharded_kvstore_standalone(output_path_param, "by_id", normalized_sharding_config)
            )
            
            total_written = 0
            
            # Write all annotations in a single transaction
            # TensorStore's sharded kvstore routes each write to the correct shard file automatically
            # After shuffle, each shard only appears in one partition, eliminating contention
            with ts.Transaction() as txn:
                kvstore_txn = by_id_kvstore.with_transaction(txn)
                
                for ann in partition:
                    encoded = encoder.encode_single_with_relationships(ann)
                    key_bytes = struct.pack(">Q", ann.id)
                    kvstore_txn[key_bytes] = encoded
                    total_written += 1
            
            return total_written
        
        # Process each partition independently
        # After shuffle by shard, each partition only writes to specific shards
        def process_partition_wrapper(partition):
            """Wrapper that returns an iterable (list) from the partition processing function."""
            count = process_partition_with_transactions(partition)
            return [count]  # Return as list so map_partitions can handle it
        
        written_counts_bag = annotations.map_partitions(process_partition_wrapper)
        
        # Sum up total annotations written (distributed sum)
        total_written = _sum_delayed_counts(written_counts_bag)
        
        step1_time = time.time() - step1_start
        logger.info(f"by_id: Processing and writing took {step1_time:.2f}s")
        
        return total_written

    def write_annotations_spatial(
        self,
        level: SpatialIndexLevel,
        cell_coords: tuple[int, ...],
        annotations: Sequence[EncodedAnnotation],
    ):
        """Write annotations for a spatial index cell.

        Args:
            level: Spatial index level
            cell_coords: Cell coordinates
            annotations: Annotations to write
        """
        encoded = self.encoder.encode_multiple(annotations)
        
        # Get or create sharded kvstore for this spatial level if needed
        spatial_key = f"spatial{level.level}"
        spatial_kvstore = self._get_sharded_kvstore("spatial", spatial_key)
        
        if "spatial" in self.sharding_config:
            # Sharded: use compressed morton code as uint64 key
            import struct
            morton_code = _zorder_compressed(cell_coords, level.grid_shape)
            # Encode morton code as big-endian uint64 bytes
            key_bytes = struct.pack(">Q", morton_code)
            spatial_kvstore.write(key_bytes, encoded).result()
        else:
            # Unsharded: use string key
            cell_key = level.cell_coords_to_key(cell_coords)
            key = f"{spatial_key}/{cell_key}"
            if isinstance(key, str):
                key_bytes = key.encode("utf-8")
            else:
                key_bytes = key
            self.base_kvstore.write(key_bytes, encoded).result()

    def write_annotations_spatial_distributed(
        self,
        cell_groups_bag: Any,  # Dask bag of (cell_coords, annotations) tuples
        level: SpatialIndexLevel,
        level_sharding_config: dict[str, Any] | None = None,
    ) -> int:
        """Write spatial annotations using distributed processing (Dask).
        
        This method writes annotations directly from a Dask bag without collecting
        them to the main process, avoiding serialization/transfer overhead.
        
        Args:
            cell_groups_bag: Dask bag of (cell_coords, list[EncodedAnnotation]) tuples
            level: Spatial index level
            
        Returns:
            Total number of annotations written
        """
        try:
            import dask.bag as db
        except ImportError:
            raise ImportError(
                "Dask is required for distributed processing. "
                "Install with: pip install dask"
            )
        
        # Convert to Dask bag if not already
        # Use larger partition size (50K) to reduce task overhead and improve worker utilization
        # Smaller partitions create too many small tasks, increasing scheduler overhead
        if not isinstance(cell_groups_bag, db.Bag):
            cell_groups_bag = db.from_sequence(cell_groups_bag, partition_size=50000)
        
        # Prepare encoder config for serialization
        encoder_config = _create_encoder_config_from_encoder(self.encoder)
        
        # Prepare sharding config (may be None)
        # Use level-specific sharding config if provided (for adaptive sharding), otherwise use base config
        if level_sharding_config is not None:
            normalized_spatial_sharding = _normalize_sharding_config_standalone(level_sharding_config)
        else:
            sharding_config = self.sharding_config if self.sharding_config else None
            normalized_spatial_sharding = None
            if sharding_config and "spatial" in sharding_config:
                normalized_spatial_sharding = _normalize_sharding_config_standalone(sharding_config["spatial"])
        output_path_param = self.output_path  # Capture for closure
        level_num_param = level.level
        grid_shape_param = level.grid_shape
        
        # OPTIMIZATION: If sharded, shuffle by shard before writing to eliminate transaction contention.
        # Check if optimization is worth it based on cells-to-shards ratio (skip if ratio < 50)
        if normalized_spatial_sharding:
            # Count cells and calculate ratio to determine if optimization is worth it
            num_cells = cell_groups_bag.count().compute()
            num_shards = 1 << normalized_spatial_sharding["shard_bits"]
            cells_to_shards_ratio = num_cells / num_shards if num_shards > 0 else float('inf')
            
            logger.info(f"spatial{level.level}: {num_cells} cells, {num_shards} shards, ratio={cells_to_shards_ratio:.2f}")
            
            if cells_to_shards_ratio < cells_to_shards_ratio_threshold:
                # Skip optimization for small ratios - use original approach
                logger.info(f"spatial{level.level}: Ratio {cells_to_shards_ratio:.2f} < {cells_to_shards_ratio_threshold}, using original shuffle approach...")
                shuffle_start = time.time()
                
                # Compute shard for each cell
                def compute_shard_for_cell(cell_item):
                    """Compute shard number for a cell - runs on worker."""
                    cell_coords, annotations = cell_item
                    morton_code = _zorder_compressed(cell_coords, grid_shape_param)
                    shard = _compute_shard_from_chunk_id(morton_code, normalized_spatial_sharding)
                    return (shard, cell_item)
                
                # Group by shard - this shuffle ensures all cells for a shard are together
                cell_groups_bag = cell_groups_bag.map(compute_shard_for_cell).groupby(lambda x: x[0])
                
                # Extract cells from groupby result and flatten
                def extract_cells_from_shard_group(shard_group):
                    """Extract cells from groupby result - runs on worker."""
                    shard, items = shard_group
                    # items is list of (shard, (cell_coords, annotations)) tuples
                    # Extract just the cell items and return as list for flattening
                    return [(cell_coords, anns) for _, (cell_coords, anns) in items]
                
                # Extract cells and flatten - after shuffle, cells for each shard are on the same partition
                cell_groups_bag = cell_groups_bag.map(extract_cells_from_shard_group).flatten()
                
                shuffle_time = time.time() - shuffle_start
                logger.info(f"spatial{level.level}: Original shuffle by shard took {shuffle_time:.2f}s")
            else:
                # Use optimized approach: local grouping, then global shuffle of groups
                logger.info(f"spatial{level.level}: Step 1 - Grouping cells by shard locally within partitions...")
                step1_start = time.time()
                
                def group_cells_by_shard_locally(partition):
                    """Group cell groups by shard within a partition - runs on worker.
                    
                    Returns:
                        List of ((shard,), [(cell_coords, annotations), ...]) tuples, wrapped in a list
                    """
                    from collections import defaultdict
                    shard_groups = defaultdict(list)
                    for cell_item in partition:
                        cell_coords, annotations = cell_item
                        morton_code = _zorder_compressed(cell_coords, grid_shape_param)
                        shard = _compute_shard_from_chunk_id(morton_code, normalized_spatial_sharding)
                        shard_groups[shard].append(cell_item)
                    # Return list of ((shard,), [cell_items]) tuples, wrapped in list (use tuple key like by_id)
                    return [[((shard,), cell_items) for shard, cell_items in shard_groups.items() if cell_items]]
                
                # Map partitions to get locally grouped shard groups
                cell_shard_groups_partitions = cell_groups_bag.map_partitions(group_cells_by_shard_locally)
                
                step1_time = time.time() - step1_start
                logger.info(f"spatial{level.level}: Step 1 (local grouping) took {step1_time:.2f}s")
                
                # Step 2: Extract groups from wrapped format, shuffle globally by shard, merge groups
                logger.info(f"spatial{level.level}: Step 2 - Extracting groups and shuffling globally...")
                step2_start = time.time()
                
                def extract_cell_shard_groups_from_partition(wrapped_result):
                    """Extract list of ((shard,), [cell_items]) tuples from wrapped partition result."""
                    # wrapped_result is [[((shard,), [cell_items]), ...]] - unwrap one level
                    if isinstance(wrapped_result, list) and wrapped_result:
                        inner_list = wrapped_result[0] if isinstance(wrapped_result[0], list) else wrapped_result
                        # Already using tuple key format, just return as-is
                        return list(inner_list)
                    return []
                
                # Extract groups and convert to bag - use tuple key (shard,) instead of int
                cell_shard_groups_bag = cell_shard_groups_partitions.map(extract_cell_shard_groups_from_partition).flatten()
                
                # Group by shard globally - now shuffling groups instead of individual cell groups!
                cell_shard_groups_grouped = cell_shard_groups_bag.groupby(lambda x: x[0])  # Group by (shard,) tuple key
                
                # Merge cell lists from groupby result
                def merge_cell_shard_groups(shard_group):
                    """Merge cell lists from groupby result - runs on worker."""
                    (shard,), items = shard_group  # items is list of ((shard,), [cell_items]) tuples
                    # Merge all cell lists into one
                    all_cells = []
                    for _, cell_items_list in items:
                        all_cells.extend(cell_items_list)
                    return ((shard,), all_cells)  # Return with tuple key for consistency
                
                # Merge groups - result is bag of ((shard,), [all_cells_for_shard]) tuples
                cell_shard_groups_merged = cell_shard_groups_grouped.map(merge_cell_shard_groups)
                
                step2_time = time.time() - step2_start
                logger.info(f"spatial{level.level}: Step 2 (global grouping and merging) took {step2_time:.2f}s")
                
                # Step 3: Extract cell groups from merged groups and flatten
                logger.info(f"spatial{level.level}: Step 3 - Extracting cell groups from merged groups...")
                step3_start = time.time()
                
                def extract_cells_from_merged_group(merged_group_item):
                    """Extract cell list from merged group item - runs on worker."""
                    (shard,), cells = merged_group_item  # cells is already a list of (cell_coords, annotations) tuples
                    return cells  # Return the list directly
                
                # Extract cell lists and flatten to individual cell groups
                # After this, cell groups are partitioned by shard (all cells for a shard are together)
                cell_groups_bag = cell_shard_groups_merged.map(extract_cells_from_merged_group).flatten()
                
                step3_time = time.time() - step3_start
                logger.info(f"spatial{level.level}: Step 3 (extract and flatten) took {step3_time:.2f}s")
        
        # OPTIMIZED APPROACH: Process partitions locally, write each shard in a transaction.
        # After shuffling by shard (if sharded), each partition only contains cells for specific shards,
        # eliminating transaction contention.
        logger.info(f"spatial{level.level}: Processing partitions with transactional writes...")
        step1_start = time.time()
        
        def process_spatial_partition_with_transactions(partition):
            """Process a partition: write each shard group in a transaction.
            
            This function runs on each worker and processes an entire partition.
            After shuffling by shard (if sharded), each partition only contains cells for
            specific shards, so we can write each shard in a single transaction without contention.
            
            Returns:
                Total number of annotations written from this partition
            """
            
            # Create encoder on worker
            encoder = _create_encoder_from_config_standalone(encoder_config)
            
            # Get or create base kvstore (cached per worker)
            base_cache_key = f"base_{output_path_param}"
            base_kvstore = _get_cached_kvstore(
                base_cache_key,
                lambda: _create_base_kvstore_standalone(output_path_param)
            )
            
            spatial_key = f"spatial{level_num_param}"
            total_written = 0
            
            if normalized_spatial_sharding:
                # Sharded: get or create sharded kvstore (cached per worker)
                # TensorStore will automatically route writes to the correct shard files
                spatial_cache_key = f"{spatial_key}_{output_path_param}"
                spatial_kvstore = _get_cached_kvstore(
                    spatial_cache_key,
                    lambda: _create_sharded_kvstore_standalone(output_path_param, spatial_key, normalized_spatial_sharding)
                )
                
                # Write all cells in a single transaction
                # TensorStore's sharded kvstore routes each write to the correct shard file automatically
                # After shuffle, each shard only appears in one partition, eliminating contention
                with ts.Transaction() as txn:
                    kvstore_txn = spatial_kvstore.with_transaction(txn)
                    
                    for cell_coords, annotations in partition:
                        encoded = encoder.encode_multiple(annotations)
                        morton_code = _zorder_compressed(cell_coords, grid_shape_param)
                        key_bytes = struct.pack(">Q", morton_code)
                        kvstore_txn[key_bytes] = encoded
                        total_written += len(annotations)
            else:
                # Unsharded: batch all cells in a single transaction per partition
                with ts.Transaction() as txn:
                    kvstore_txn = base_kvstore.with_transaction(txn)
                    
                    for cell_coords, annotations in partition:
                        encoded = encoder.encode_multiple(annotations)
                        cell_key_parts = [str(c) for c in cell_coords]
                        cell_key = "_".join(cell_key_parts)
                        key = f"{spatial_key}/{cell_key}"
                        key_bytes = key.encode("utf-8")
                        kvstore_txn[key_bytes] = encoded
                        total_written += len(annotations)
            
            return total_written
        
        # Process each partition independently
        # After shuffle by shard (if sharded), each partition only writes to specific shards
        def process_partition_wrapper(partition):
            """Wrapper that returns an iterable (list) from the partition processing function."""
            count = process_spatial_partition_with_transactions(partition)
            return [count]
        
        written_counts_bag = cell_groups_bag.map_partitions(process_partition_wrapper)
        
        # Sum up total annotations written (distributed sum)
        total_written = _sum_delayed_counts(written_counts_bag)
        
        step1_time = time.time() - step1_start
        logger.info(f"spatial{level.level}: Processing and writing took {step1_time:.2f}s")
        
        return total_written

    def write_annotations_relationship(
        self,
        relationship: str,
        object_id: int,
        annotations: Sequence[EncodedAnnotation],
    ):
        """Write annotations for a relationship index.

        Args:
            relationship: Relationship name
            object_id: Related object id
            annotations: Annotations to write
        """
        encoded = self.encoder.encode_multiple(annotations)
        
        # Get or create sharded kvstore for this relationship if needed
        rel_key = f"rel_{relationship}"
        rel_kvstore = self._get_sharded_kvstore(f"rel_{relationship}", rel_key)
        
        rel_config = self.sharding_config.get("relationships", {}).get(relationship)
        if rel_config:
            # Sharded: use uint64 key as bytes
            import struct
            key_bytes = struct.pack(">Q", object_id)
            rel_kvstore.write(key_bytes, encoded).result()
        else:
            # Unsharded: use string key
            key = f"{rel_key}/{object_id}"
            if isinstance(key, str):
                key_bytes = key.encode("utf-8")
            else:
                key_bytes = key
            self.base_kvstore.write(key_bytes, encoded).result()
    
    def write_annotations_relationship_distributed(
        self,
        relationship_groups_bag: Any,  # dask.bag.Bag of ((relationship, object_id), annotations) tuples
    ) -> int:
        """Write relationship indices using distributed processing (Dask).
        
        This method writes relationship indices directly from a bag without collecting
        all annotations to the main process, avoiding serialization/transfer overhead.
        
        Args:
            relationship_groups_bag: Dask bag of ((relationship_name, object_id), annotations) tuples
            
        Returns:
            Total number of relationship groups written
        """
        try:
            import dask.bag as db
        except ImportError:
            raise ImportError(
                "Dask is required for distributed processing. "
                "Install with: pip install dask"
            )
        
        # relationship_groups_bag is already a bag, use it directly
        rel_groups_bag = relationship_groups_bag
        
        # Prepare encoder config for serialization
        encoder_config = _create_encoder_config_from_encoder(self.encoder)
        
        # Prepare sharding config (may be None)
        sharding_config = self.sharding_config if self.sharding_config else None
        
        output_path_param = self.output_path  # Capture for closure
        
        # OPTIMIZATION: If any relationships are sharded, shuffle by (relationship, shard) before writing
        # to eliminate transaction contention. This ensures all groups for a (relationship, shard) are
        # on the same partition, so each shard is written in a single transaction.
        if sharding_config and sharding_config.get("relationships"):
            # Check if any relationships are sharded
            has_sharded_relationships = any(
                sharding_config.get("relationships", {}).get(rel) is not None
                for rel in sharding_config.get("relationships", {}).keys()
            )
            
            if has_sharded_relationships:
                logger.info("relationships: Shuffling sharded relationships by (relationship, shard) to eliminate transaction contention...")
                shuffle_start = time.time()
                
                # OPTIMIZATION: Pre-normalize sharding configs for each relationship type (once per relationship type)
                # instead of normalizing in the map function (once per relationship group, which could be 50k+ times)
                normalized_rel_configs = {}
                for rel_name, rel_config in (sharding_config.get("relationships", {})).items():
                    if rel_config:
                        normalized_rel_configs[rel_name] = _normalize_sharding_config_standalone(rel_config)
                
                # Compute (relationship, shard) key for each group
                def compute_relationship_shard_key(item):
                    """Compute (relationship, shard) key for a relationship group - runs on worker."""
                    ((relationship, object_id), annotations) = item
                    normalized = normalized_rel_configs.get(relationship)
                    if normalized:
                        # Sharded: compute shard number using pre-normalized config
                        shard = _compute_shard_from_chunk_id(object_id, normalized)
                        return ((relationship, shard), item)
                    else:
                        # Unsharded: use None as shard (will not be shuffled)
                        return ((relationship, None), item)
                
                # Group by (relationship, shard) - this shuffle ensures all groups for a (relationship, shard) are together
                # Note: Unsharded relationships (shard=None) will also be grouped, but that's fine
                rel_groups_bag = rel_groups_bag.map(compute_relationship_shard_key).groupby(lambda x: x[0])
                
                # Extract groups from groupby result
                # groupby returns ((relationship, shard), [((relationship, shard), item), ...])
                def extract_groups_from_key_group(key_group):
                    """Extract relationship groups from groupby result - runs on worker."""
                    (relationship, shard), items = key_group
                    # items is list of ((relationship, shard), ((relationship, object_id), annotations)) tuples
                    # Extract just the relationship group items and return as list for flattening
                    return [item for _, item in items]
                
                # Extract groups and flatten - after shuffle, groups for each (relationship, shard) are on the same partition
                rel_groups_bag = rel_groups_bag.map(extract_groups_from_key_group).flatten()
                
                shuffle_time = time.time() - shuffle_start
                logger.info(f"relationships: Shuffle by (relationship, shard) took {shuffle_time:.2f}s")
        
        # OPTIMIZED APPROACH: Process partitions locally, write each (relationship, shard) group in a transaction.
        # After shuffling by (relationship, shard) (if any relationships are sharded), each partition only contains
        # groups for specific (relationship, shard) combinations, eliminating transaction contention.
        logger.info("relationships: Processing partitions with transactional writes...")
        step1_start = time.time()
        
        def process_relationship_partition_with_transactions(partition):
            """Process a partition: write each (relationship, shard) group in a transaction.
            
            This function runs on each worker and processes an entire partition.
            After shuffling by (relationship, shard) (if any relationships are sharded), each partition
            only contains groups for specific (relationship, shard) combinations, so we can write each
            shard in a single transaction without contention.
            
            Returns:
                Total number of annotations written from this partition
            """
            # Create encoder on worker
            encoder = _create_encoder_from_config_standalone(encoder_config)
            
            # Get or create base kvstore (cached per worker)
            base_cache_key = f"base_{output_path_param}"
            base_kvstore = _get_cached_kvstore(
                base_cache_key,
                lambda: _create_base_kvstore_standalone(output_path_param)
            )
            
            total_written = 0
            
            # Group by relationship (each relationship has its own kvstore)
            # TensorStore will automatically route writes to the correct shard files for sharded relationships
            relationship_groups = defaultdict(list)
            for (relationship, object_id), annotations in partition:
                relationship_groups[relationship].append((object_id, annotations))
            
            # Write each relationship's groups in a single transaction
            # For sharded relationships, TensorStore routes writes to correct shard files automatically
            # After shuffle, each (relationship, shard) only appears in one partition, eliminating contention
            for relationship, groups in relationship_groups.items():
                rel_key = f"rel_{relationship}"
                rel_config = sharding_config.get("relationships", {}).get(relationship) if sharding_config else None
                
                if rel_config:
                    # Sharded: get kvstore for this relationship
                    normalized = _normalize_sharding_config_standalone(rel_config)
                    
                    rel_cache_key = f"{rel_key}_{output_path_param}"
                    rel_kvstore = _get_cached_kvstore(
                        rel_cache_key,
                        lambda: _create_sharded_kvstore_standalone(output_path_param, rel_key, normalized)
                    )
                    
                    # Write all groups for this relationship in a single transaction
                    # TensorStore routes each write to the correct shard file automatically
                    with ts.Transaction() as txn:
                        kvstore_txn = rel_kvstore.with_transaction(txn)
                        
                        for object_id, annotations in groups:
                            encoded = encoder.encode_multiple(annotations)
                            key_bytes = struct.pack(">Q", object_id)
                            kvstore_txn[key_bytes] = encoded
                            total_written += len(annotations)
                else:
                    # Unsharded: write to base kvstore
                    with ts.Transaction() as txn:
                        kvstore_txn = base_kvstore.with_transaction(txn)
                        
                        for object_id, annotations in groups:
                            encoded = encoder.encode_multiple(annotations)
                            key = f"{rel_key}/{object_id}"
                            key_bytes = key.encode("utf-8")
                            kvstore_txn[key_bytes] = encoded
                            total_written += len(annotations)
            
            return total_written
        
        # Process each partition independently
        # After shuffle by (relationship, shard) (if any relationships are sharded), each partition only writes to specific shards
        def process_partition_wrapper(partition):
            """Wrapper that returns an iterable (list) from the partition processing function."""
            count = process_relationship_partition_with_transactions(partition)
            return [count]
        
        written_counts_bag = rel_groups_bag.map_partitions(process_partition_wrapper)
        
        # Sum up total annotations written (distributed sum)
        total_written = _sum_delayed_counts(written_counts_bag)
        
        step1_time = time.time() - step1_start
        logger.info(f"relationships: Processing and writing took {step1_time:.2f}s")
        
        return total_written


    def _get_spatial_limit(self) -> int:
        """Get spatial index limit from config."""
        return self.spatial_limit

