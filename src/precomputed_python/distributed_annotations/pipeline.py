"""Distributed annotation pipeline for generating precomputed annotation datasets.

This module provides a distributed framework for generating very large-scale
(billions of rows) Neuroglancer precomputed annotation datasets using Dask for
parallel processing and TensorStore for cloud storage.
"""

import functools
import logging
import math
import time
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

try:
    import dask
    import dask.bag as db
    import dask.distributed as dd
except ImportError:
    # Dask is optional - only needed when use_dask=True
    dask = None
    db = None
    dd = None

from neuroglancer import coordinate_space, viewer_state

from .encoder import AnnotationEncoder, EncodedAnnotation
from .spatial_index import SpatialIndexBuilder, SpatialIndexLevel
from .writer import DistributedAnnotationWriter

logger = logging.getLogger(__name__)


class _PartitionEncodingResult(NamedTuple):
    """Result from encoding a partition."""
    encoded: list[EncodedAnnotation]
    local_lower: list[float]
    local_upper: list[float]


def _encode_partition_standalone(
    partition: list[tuple[Any, ...]],
    annotation_type: str,
    rank: int,
    properties: list[dict[str, Any]],
) -> list[_PartitionEncodingResult]:
    """Standalone encoding function that can be pickled for Dask.
    
    Creates an encoder on-demand from config parameters.
    
    Args:
        partition: List of (id, geometry, properties, relationships) tuples
        annotation_type: Type of annotation
        rank: Number of spatial dimensions
        properties: Property specifications as plain dicts (fully serializable)
        
    Returns:
        List containing a single _PartitionEncodingResult (wrapped for map_partitions)
    """
    # Convert property dicts back to AnnotationPropertySpec objects
    # (needed for encoder, but only created on worker, not serialized)
    from neuroglancer import viewer_state
    property_specs = [
        viewer_state.AnnotationPropertySpec(prop_dict)
        for prop_dict in properties
    ]
    
    # Create encoder on-demand (fast, no TensorStore dependencies)
    encoder = AnnotationEncoder(
        annotation_type=annotation_type,
        rank=rank,
        properties=property_specs,
    )
    
    encoded = []
    local_lower = [float("inf")] * rank
    local_upper = [float("-inf")] * rank
    
    for item in partition:
        if len(item) < 2:
            raise ValueError(f"Annotation tuple must have at least (id, geometry), got {len(item)} items")
        
        ann_id = item[0]
        geometry = item[1]
        properties_dict = item[2] if len(item) > 2 else {}
        relationships = item[3] if len(item) > 3 else []
        
        encoded_ann = encoder.encode_single(
            annotation_id=ann_id,
            geometry=geometry,
            properties=properties_dict,
            relationships=relationships,
        )
        
        encoded.append(encoded_ann)
        
        # Update local bounds
        for d, (min_val, max_val) in enumerate(encoded_ann.bounding_box):
            local_lower[d] = min(local_lower[d], min_val)
            local_upper[d] = max(local_upper[d], max_val)
    
    # Return as a list containing the NamedTuple so map_partitions treats it as a single item
    # This avoids Dask iterating over the NamedTuple fields
    return [_PartitionEncodingResult(encoded=encoded, local_lower=local_lower, local_upper=local_upper)]


class _EncodePartitionCallable:
    """Callable class that can be pickled for Dask.
    
    This wraps the encoding function with config parameters.
    Properties are stored as plain dicts to ensure full serializability.
    """
    def __init__(self, annotation_type: str, rank: int, properties: list[dict[str, Any]]):
        self.annotation_type = annotation_type
        self.rank = rank
        self.properties = properties
    
    def __call__(self, partition: list[tuple[Any, ...]]) -> list[_PartitionEncodingResult]:
        """Call the encoding function with stored config."""
        return _encode_partition_standalone(
            partition, self.annotation_type, self.rank, self.properties
        )


def _get_encoded_length(result: _PartitionEncodingResult) -> int:
    """Get the length of encoded annotations list."""
    return len(result.encoded)


def _get_encoded_list(result: _PartitionEncodingResult) -> list[EncodedAnnotation]:
    """Get the encoded annotations list."""
    return result.encoded


def _get_first_element(item: tuple) -> Any:
    """Get the first element of a tuple (for groupby key)."""
    return item[0]


def _merge_relationship_indices_standalone(acc: dict, partition_result: Any) -> dict:
    """Merge relationship indices from partition results into accumulator.
    
    This is used with fold() to merge partition-level relationship indices without shuffling.
    Uses tree reduction instead of hash partitioning, avoiding network shuffle.
    
    Args:
        acc: Accumulator dict mapping rel_name -> {obj_id: [annotations]}
        partition_result: Partition result which is a dict mapping rel_name -> {obj_id: [annotations]}
                         or wrapped in a list
        
    Returns:
        Merged dict mapping rel_name -> {obj_id: [annotations]} (concatenated)
    """
    # Initialize accumulator if needed
    if acc is None or not isinstance(acc, dict):
        acc = {}
    
    # Handle different input formats
    if isinstance(partition_result, list):
        if not partition_result:
            return acc
        partition_result = partition_result[0] if isinstance(partition_result[0], dict) else {}
    elif isinstance(partition_result, tuple):
        # Dask might wrap results in tuples
        if len(partition_result) == 1 and isinstance(partition_result[0], dict):
            partition_result = partition_result[0]
        else:
            return acc  # Skip unexpected formats
    
    if not isinstance(partition_result, dict):
        return acc
    
    # Merge relationship indices
    for rel_name, obj_indices in partition_result.items():
        if rel_name not in acc:
            acc[rel_name] = {}
        for obj_id, anns in obj_indices.items():
            if obj_id not in acc[rel_name]:
                acc[rel_name][obj_id] = []
            acc[rel_name][obj_id].extend(anns)
    
    return acc


def _merge_emitted_dicts_standalone(acc: dict, partition_result: Any) -> dict:
    """Merge emitted annotations from partition results into accumulator.
    
    This is used with fold() to merge partition-level dicts without shuffling.
    Uses tree reduction instead of hash partitioning, avoiding network shuffle.
    
    Args:
        acc: Accumulator dict mapping cell_coords -> list of annotations
        partition_result: Partition result which can be:
            - Dict with 'emitted' key: {'emitted': {cell_coords: [anns]}, 'remaining': {...}}
            - List containing such a dict: [{'emitted': {...}, 'remaining': {...}}]
            - Tuple (if Dask wraps it)
            
    Returns:
        Merged dict mapping cell_coords -> list of annotations (concatenated)
    """
    # Initialize accumulator if needed
    if acc is None or not isinstance(acc, dict):
        acc = defaultdict(list)
    
    # Handle different input formats
    if isinstance(partition_result, list):
        if not partition_result:
            return acc
        partition_result = partition_result[0] if isinstance(partition_result[0], dict) else {}
    elif isinstance(partition_result, tuple):
        # Dask might wrap results in tuples
        if len(partition_result) == 1 and isinstance(partition_result[0], dict):
            partition_result = partition_result[0]
        else:
            return acc  # Skip unexpected formats
    
    if not isinstance(partition_result, dict):
        return acc
    
    # Extract emitted dict
    emitted_dict = partition_result.get('emitted', {})
    
    # Merge into accumulator
    for cell_coords, anns in emitted_dict.items():
        if cell_coords not in acc:
            acc[cell_coords] = []
        acc[cell_coords].extend(anns)
    
    return acc


def _get_second_element(item: tuple) -> Any:
    """Get the second element of a tuple."""
    return item[1]


class _MapToCellsCallable:
    """Callable class to map annotations to spatial cells.
    
    This can be pickled for Dask by storing only serializable config.
    """
    def __init__(
        self,
        rank: int,
        lower_bound: list[float],
        chunk_size: list[float],
        grid_shape: tuple[int, ...],
    ):
        self.rank = rank
        self.lower_bound = lower_bound
        self.chunk_size = chunk_size
        self.grid_shape = grid_shape
    
    def __call__(self, ann: EncodedAnnotation) -> list[tuple[tuple[int, ...], EncodedAnnotation]]:
        """Map annotation to all spatial cells it intersects."""
        from .spatial_index import intersects_cell
        
        # Compute cell range from annotation bounding box
        ann_bbox = ann.bounding_box
        cell_ranges = []
        for d in range(self.rank):
            ann_min, ann_max = ann_bbox[d]
            # Convert to cell coordinates
            cell_min = int((ann_min - self.lower_bound[d]) / self.chunk_size[d])
            cell_max = int((ann_max - self.lower_bound[d]) / self.chunk_size[d])
            # Clamp to valid range
            cell_min = max(0, cell_min)
            cell_max = min(self.grid_shape[d] - 1, cell_max)
            cell_ranges.append((cell_min, cell_max))
        
        # Generate all cells in the range and check intersection
        results = []
        
        def generate_cells(coords: list[int], dim: int):
            if dim == self.rank:
                cell_coords = tuple(coords)
                # Compute cell bounds manually (can't use level.get_cell_bounds)
                cell_bounds = []
                for d in range(self.rank):
                    cell_min = self.lower_bound[d] + cell_coords[d] * self.chunk_size[d]
                    cell_max = cell_min + self.chunk_size[d]
                    cell_bounds.append((cell_min, cell_max))
                cell_bounds = tuple(cell_bounds)
                
                if intersects_cell(ann, cell_bounds):
                    results.append((cell_coords, ann))
                return
            for c in range(cell_ranges[dim][0], cell_ranges[dim][1] + 1):
                generate_cells(coords + [c], dim + 1)
        
        generate_cells([], 0)
        
        # If no intersection (shouldn't happen), put in root cell
        if not results:
            root_cell = tuple([0] * self.rank)
            results.append((root_cell, ann))
        
        return results


def _count_cell_standalone(group_item: tuple[tuple[int, ...], list]) -> tuple[tuple[int, ...], int]:
    """Count annotations in a cell group (module-level function)."""
    cell_coords, items = group_item
    return (cell_coords, len(items))


class _SampleCellGroupCallable:
    """Callable class to sample annotations in a cell group.
    
    This can be pickled for Dask by storing only serializable config.
    """
    def __init__(self, probability: float, seed: int):
        self.probability = probability
        self.seed = seed
    
    def __call__(
        self, group_item: tuple[tuple[int, ...], list]
    ) -> tuple[tuple[int, ...], list[EncodedAnnotation], list[EncodedAnnotation]]:
        """Sample annotations in a cell group."""
        cell_coords, items = group_item
        annotations = [item[1] for item in items]  # Extract annotations from (cell_coords, ann) tuples
        
        # Sample with fixed seed for reproducibility
        random.seed(self.seed + hash(cell_coords))
        emitted = []
        remaining = []
        for ann in annotations:
            if random.random() < self.probability:
                emitted.append(ann)
            else:
                remaining.append(ann)
        
        return (cell_coords, emitted, remaining)


class _TakeFirstNCallable:
    """Callable class to take first N annotations from a cell group.
    
    This is more efficient than probabilistic sampling as it avoids
    needing to compute maxCount. Annotations should be pre-randomized.
    """
    def __init__(self, limit: int):
        self.limit = limit
    
    def __call__(
        self, group_item: tuple[tuple[int, ...], list]
    ) -> tuple[tuple[int, ...], list[EncodedAnnotation], list[EncodedAnnotation]]:
        """Take first N annotations from a cell group.
        
        Args:
            group_item: (cell_coords, [(cell_coords, ann), ...]) from Dask groupby
        
        Returns:
            (cell_coords, emitted_annotations, remaining_annotations)
        """
        cell_coords, items = group_item
        annotations = [item[1] for item in items]  # Extract annotations from (cell_coords, ann) tuples
        
        # Simply take first N (annotations should be pre-randomized)
        emitted = annotations[:self.limit]
        remaining = annotations[self.limit:]
        
        return (cell_coords, emitted, remaining)




class _MapRemainingToChildCellsCallable:
    """Callable class to map remaining annotations from parent cells to child cells."""
    def __init__(
        self,
        rank: int,
        next_lower_bound: list[float],
        next_chunk_size: list[float],
        next_grid_shape: tuple[int, ...],
    ):
        self.rank = rank
        self.next_lower_bound = next_lower_bound
        self.next_chunk_size = next_chunk_size
        self.next_grid_shape = next_grid_shape
    
    def __call__(
        self, item: tuple[tuple[int, ...], EncodedAnnotation]
    ) -> list[tuple[tuple[int, ...], EncodedAnnotation]]:
        """Map a remaining annotation from parent cell to child cells.
        
        Args:
            item: (parent_cell_coords, ann) tuple
            
        Returns:
            List of (child_cell_coords, ann) tuples for intersecting child cells
        """
        from .spatial_index import intersects_cell
        
        parent_cell_coords, ann = item
        
        # Compute child cells for this parent cell
        # Each dimension is divided by 2, so child cells are in range [2*coord, 2*coord+1]
        child_cells = []
        
        def generate_child_cells(coords: list[int], dim: int):
            if dim == self.rank:
                child_cells.append(tuple(coords))
                return
            parent_coord = parent_cell_coords[dim]
            for offset in [0, 1]:
                child_coord = 2 * parent_coord + offset
                if child_coord < self.next_grid_shape[dim]:
                    generate_child_cells(coords + [child_coord], dim + 1)
        
        generate_child_cells([], 0)
        
        # Check which child cells the annotation intersects
        results = []
        for child_cell in child_cells:
            child_bounds = tuple(
                (self.next_lower_bound[d] + child_cell[d] * self.next_chunk_size[d],
                 self.next_lower_bound[d] + (child_cell[d] + 1) * self.next_chunk_size[d])
                for d in range(self.rank)
            )
            if intersects_cell(ann, child_bounds):
                results.append((child_cell, ann))
        
        return results


class _ExtractRemainingForNextLevelCallable:
    """Callable class to extract remaining annotations and map to child cells.
    
    This processes remaining annotations in a distributed way without
    collecting all data to the main process.
    """
    def __init__(
        self,
        current_level: int,
        rank: int,
        current_lower_bound: list[float],
        current_chunk_size: list[float],
        current_grid_shape: tuple[int, ...],
        next_lower_bound: list[float],
        next_chunk_size: list[float],
        next_grid_shape: tuple[int, ...],
    ):
        self.current_level = current_level
        self.rank = rank
        self.current_lower_bound = current_lower_bound
        self.current_chunk_size = current_chunk_size
        self.current_grid_shape = current_grid_shape
        self.next_lower_bound = next_lower_bound
        self.next_chunk_size = next_chunk_size
        self.next_grid_shape = next_grid_shape
    
    def _get_child_cells(self, cell_coords: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Compute child cells for a given parent cell."""
        # Child cells are cells in the next level that are contained in this parent cell
        # Each dimension is divided by 2, so child cells are in range [2*coord, 2*coord+1]
        child_cells = []
        
        def generate_child_cells(coords: list[int], dim: int):
            if dim == self.rank:
                child_cells.append(tuple(coords))
                return
            # Each parent cell maps to 2 child cells per dimension
            parent_coord = cell_coords[dim]
            for offset in [0, 1]:
                child_coord = 2 * parent_coord + offset
                if child_coord < self.next_grid_shape[dim]:
                    generate_child_cells(coords + [child_coord], dim + 1)
        
        generate_child_cells([], 0)
        return child_cells
    
    def _get_cell_bounds(self, cell_coords: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
        """Compute cell bounds for given coordinates."""
        bounds = []
        for d in range(self.rank):
            cell_min = self.next_lower_bound[d] + cell_coords[d] * self.next_chunk_size[d]
            cell_max = cell_min + self.next_chunk_size[d]
            bounds.append((cell_min, cell_max))
        return tuple(bounds)
    
    def __call__(
        self, group_item: tuple[tuple[int, ...], list]
    ) -> list[tuple[tuple[int, ...], EncodedAnnotation]]:
        """Extract remaining annotations and map to child cells."""
        from .spatial_index import intersects_cell
        
        cell_coords, emitted, remaining = group_item
        if not remaining:
            return []
        
        # Get child cells for this parent cell
        try:
            child_cells = self._get_child_cells(cell_coords)
            
            results = []
            for child_cell in child_cells:
                child_bounds = self._get_cell_bounds(child_cell)
                for ann in remaining:
                    if intersects_cell(ann, child_bounds):
                        results.append((child_cell, ann))
            return results
        except Exception:
            # Can't compute child cells - discard remaining
            return []


def _extract_emitted_standalone(group_item: tuple[tuple[int, ...], list, list]) -> tuple[tuple[int, ...], list[EncodedAnnotation]]:
    """Extract only emitted annotations from cell group (module-level function).
    
    Args:
        group_item: (cell_coords, emitted_annotations, remaining_annotations) from _SortAndTakeNCallable
        
    Returns:
        (cell_coords, emitted_annotations)
    """
    cell_coords, emitted, remaining = group_item
    return (cell_coords, emitted)


def _extract_cell_group_standalone(group_item: tuple[tuple[int, ...], list]) -> tuple[tuple[int, ...], list]:
    """Extract cell group from Dask groupby result for writing.
    
    Args:
        group_item: (cell_coords, [(cell_coords, emitted_annotations), ...]) from Dask groupby
        
    Returns:
        (cell_coords, all_annotations) - combined annotations for this cell
    """
    cell_coords, items = group_item
    # items is a list of (cell_coords, emitted_annotations) tuples
    # All have the same cell_coords, so we combine all annotations
    all_annotations = []
    for _, anns in items:
        if anns:  # Only add non-empty lists
            all_annotations.extend(anns)
    return (cell_coords, all_annotations)


class _LocalGroupAndTakeNCallable:
    """Callable class to group by cell locally within a partition, sort, and take first N.
    
    This avoids the expensive global merge by taking first N per partition per cell.
    The limit per partition is limit / num_partitions, so total across partitions
    will be approximately the desired limit (may be slightly less if some partitions
    don't have enough annotations for a cell).
    """
    def __init__(self, limit_per_partition: int):
        self.limit_per_partition = limit_per_partition
    
    def __call__(self, partition: list[tuple[tuple[int, ...], EncodedAnnotation]]) -> list[dict]:
        """Group by cell_coords locally and take first N per cell.
        
        IMPORTANT: Assumes input data is pre-randomized. If partitions have spatial biases,
        the spatial index will be unbalanced. See documentation for data source requirements.
        
        Args:
            partition: List of (cell_coords, ann) tuples from a single partition
            
        Returns:
            List containing a dict with two keys:
            - 'emitted': dict mapping cell_coords -> list of annotations (first N)
            - 'remaining': dict mapping cell_coords -> list of annotations (rest)
            (Wrapped in list to prevent Dask from flattening the dict)
        """
        # Handle empty partition
        if not partition:
            return [{'emitted': {}, 'remaining': {}}]
        
        # Group by cell_coords within this partition (local, no network shuffle)
        cell_groups = defaultdict(list)
        for cell_coords, ann in partition:
            cell_groups[cell_coords].append(ann)
        
        # For each cell, take first N (assumes pre-randomized input)
        emitted = {}
        remaining = {}
        for cell_coords, anns in cell_groups.items():
            # Take first N for emitted, rest for remaining
            emitted[cell_coords] = anns[:self.limit_per_partition]
            if len(anns) > self.limit_per_partition:
                remaining[cell_coords] = anns[self.limit_per_partition:]
        
        # Wrap in list to prevent Dask from flattening the dict (dicts are iterable over keys)
        return [{'emitted': emitted, 'remaining': remaining}]


class _MergePartitionResultsCallable:
    """Callable class to merge partition results and take first N globally.
    
    This combines results from multiple partitions and takes first N per cell.
    Also tracks remaining annotations for propagation to next level.
    """
    def __init__(self, limit: int):
        self.limit = limit
    
    def __call__(self, acc: dict, partition_result: dict | tuple) -> dict:
        """Merge partition result into accumulator.
        
        Args:
            acc: Accumulator dict mapping cell_coords -> list of annotations
            partition_result: Dict from one partition mapping cell_coords -> list of annotations
                             OR tuple if Dask wraps it
        
        Returns:
            Merged dict with all items per cell (will be trimmed to first N in final step)
        """
        # Initialize accumulator if needed
        if acc is None or not isinstance(acc, dict):
            acc = defaultdict(list)
        
        # Handle case where Dask wraps the result in a tuple
        if isinstance(partition_result, tuple):
            # If it's a tuple, try to extract the dict
            # Dask might wrap single-element results
            if len(partition_result) == 1 and isinstance(partition_result[0], dict):
                partition_result = partition_result[0]
            else:
                # Unexpected format, skip this partition
                logger.warning(f"Unexpected partition_result format: {type(partition_result)}, skipping")
                return acc
        
        # Merge partition results
        if isinstance(partition_result, dict):
            for cell_coords, items in partition_result.items():
                acc[cell_coords].extend(items)
        else:
            logger.warning(f"partition_result is not a dict: {type(partition_result)}, skipping")
        
        return acc


def _finalize_merged_results(
    merged_result: dict, limit: int
) -> tuple[dict[tuple[int, ...], list[EncodedAnnotation]], dict[tuple[int, ...], list[EncodedAnnotation]]]:
    """Finalize merged results by taking first N per cell.
    
    Args:
        merged_result: Dict mapping cell_coords -> list of annotations (all items)
        limit: Maximum annotations per cell
        
    Returns:
        Tuple of (emitted_dict, remaining_dict) where:
        - emitted_dict: cell_coords -> list of annotations (first N)
        - remaining_dict: cell_coords -> list of annotations (rest, for next level)
    """
    emitted = {}
    remaining = {}
    
    for cell_coords, all_anns in merged_result.items():
        # Take first N for emitted, rest for remaining
        emitted[cell_coords] = all_anns[:limit]
        if len(all_anns) > limit:
            remaining[cell_coords] = all_anns[limit:]
    
    return (emitted, remaining)




class AnnotationPipeline:
    """Pipeline for generating precomputed annotation datasets.
    
    This pipeline processes annotations through multiple stages:
    1. Encoding: Convert annotations to binary format
    2. Spatial Index: Build multi-level spatial index
    3. Relationship Index: Build relationship indices
    4. Writing: Write all indices to storage
    """

    def __init__(
        self,
        output_path: str,
        coordinate_space: coordinate_space.CoordinateSpace,
        annotation_type: str,
        properties: Sequence[viewer_state.AnnotationPropertySpec] = (),
        relationships: Sequence[str] = (),
        spatial_index_config: dict[str, Any] | None = None,
        sharding_config: dict[str, Any] | None = None,
        use_dask: bool = False,
        dask_client: Any = None,  # dask.distributed.Client
    ):
        """Initialize annotation pipeline.
        
        Args:
            output_path: Path to output directory (local or gs://, s3://, etc.)
            coordinate_space: Coordinate space definition
            annotation_type: Type of annotation (point, line, etc.)
            properties: Property specifications
            relationships: Relationship type names
            spatial_index_config: Configuration for spatial index
            sharding_config: Sharding configuration for indices
            use_dask: Whether to use Dask for distributed processing
            dask_client: Dask distributed client (required if use_dask=True)
        """
        self.output_path = output_path
        self.coordinate_space = coordinate_space
        self.annotation_type = annotation_type
        self.properties = list(properties)
        self.relationships = list(relationships)
        self.use_dask = use_dask
        self.dask_client = dask_client
        self.spatial_index_config = spatial_index_config or {}

        # Extract spatial limit from config
        self.spatial_limit = self.spatial_index_config.get("limit", 1000)

        # Initialize encoder
        self.encoder = AnnotationEncoder(
            annotation_type=annotation_type,
            rank=coordinate_space.rank,
            properties=properties,
        )

        # Initialize spatial index builder (will compute bounds from data)
        # For now, we'll need to update bounds after processing annotations
        self.spatial_builder: SpatialIndexBuilder | None = None

        self.writer = DistributedAnnotationWriter(
            output_path=output_path,
            sharding_config=sharding_config or {},
            encoder_config={
                "annotation_type": self.annotation_type,
                "rank": self.coordinate_space.rank,
                "properties": [p.to_json() for p in self.properties],
            },
        )
        # Set spatial_limit on writer (needed for write_info)
        self.writer.spatial_limit = self.spatial_limit
        
        # Store level-specific sharding configs for adaptive sharding
        # Maps level number -> sharding config dict
        self.level_sharding_configs: dict[int, dict[str, Any]] = {}

    def process(
        self,
        annotation_iterator: Iterator[tuple[Any, ...]] | None = None,
        annotation_bag: Any = None,  # dask.bag.Bag | None
    ):
        """Process annotations and generate all indices.

        IMPORTANT: Input data must be pre-randomized to avoid spatial biases. If partitions
        have strong spatial biases (e.g., CSV rows in spatial order, SQL results sorted by
        location), the spatial index will be unbalanced and computation will not be well
        distributed. See DATA_SOURCE_ADAPTERS.md for guidance on ensuring proper randomization.

        Args:
            annotation_iterator: Iterator yielding (id, geometry, properties, relationships) tuples.
                                Only used if annotation_bag is None and use_dask=False.
                                WARNING: If use_dask=True, this will collect all data through main process!
                                Must be pre-randomized to avoid spatial biases.
            annotation_bag: Pre-constructed Dask bag of (id, geometry, properties, relationships) tuples.
                           Recommended for large datasets - workers read partitions directly.
                           If provided, use_dask must be True and dask_client must be set.
                           Partitions must be pre-randomized to avoid spatial biases.
        """
        pipeline_start = time.time()
        stage_times = {}
        
        # Determine which input to use
        if annotation_bag is not None:
            # Pre-constructed bag provided - use it directly (no collection!)
            if not self.use_dask or self.dask_client is None:
                raise ValueError(
                    "annotation_bag provided but use_dask=False or dask_client is None. "
                    "Distributed bags require Dask client."
                )
            annotations_bag = annotation_bag
            logger.info("Using provided Dask bag (distributed reading, no main process collection)")
        elif annotation_iterator is not None:
            # Iterator provided - convert to bag if using Dask
            if self.use_dask and self.dask_client is not None:
                if db is None:
                    raise ImportError("Dask is required for distributed processing. Install with: pip install dask")
                # Convert iterator to Dask bag for distributed processing
                # WARNING: from_sequence with iterator collects all data through main process!
                # This is only suitable for small datasets. For large datasets, use annotation_bag.
                logger.warning(
                    "Using annotation_iterator with Dask - data will be collected through main process. "
                    "For large datasets, use annotation_bag with distributed reading instead."
                )
                try:
                    # Try to get number of workers from the cluster
                    n_workers = len(self.dask_client.scheduler_info()['workers'])
                    npartitions = max(1, n_workers * 2)  # 2 partitions per worker for better load balancing
                except (KeyError, AttributeError):
                    # Fallback to a reasonable default
                    logger.warning('not able to find number of workers from cluster, defaultings to 8 partitions')
                    npartitions = 8
                
                logger.info(f"Creating Dask bag in main process from iterator with {npartitions} partitions")
                # Suppress the warning about large graph - this is expected for initial data load
                # Subsequent operations will use persisted data and won't have this issue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Sending large graph.*")
                    annotations_bag = db.from_sequence(annotation_iterator, npartitions=npartitions)
            else:
                # Not using Dask - keep as None, will use iterator directly
                annotations_bag = None
        else:
            raise ValueError("Must provide either annotation_iterator or annotation_bag")
        
        # Stage 1: Encode annotations and compute bounds
        logger.info("Stage 1: Encoding annotations and computing bounds...")
        stage_start = time.time()
        if annotations_bag is not None:
            encoded_bag, lower_bound, upper_bound, total_annotations = self._encode_annotations_distributed(annotations_bag)
        else:
            encoded_annotations, lower_bound, upper_bound = self._encode_annotations(annotation_iterator)
            total_annotations = len(encoded_annotations)
            encoded_bag = None
        stage_times["1. Encoding"] = time.time() - stage_start

        # Initialize spatial index builder with computed bounds
        initial_chunk_size = self.spatial_index_config.get(
            "initial_chunk_size",
            None,  # Not used - level 0 always uses full space
        )

        self.spatial_builder = SpatialIndexBuilder(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            initial_chunk_size=initial_chunk_size or [1.0] * self.coordinate_space.rank,  # Dummy value, not used
            limit=self.spatial_limit,
            rank=self.coordinate_space.rank,
        )

        # Stage 2: Build spatial index
        logger.info("Stage 2: Building spatial index...")
        stage_start = time.time()
        if encoded_bag is not None:
            spatial_results = self._build_spatial_index_distributed(encoded_bag, total_annotations)
        else:
            spatial_results = self._build_spatial_index(encoded_annotations)
        stage_times["2. Spatial Index"] = time.time() - stage_start

        # Stage 3: Build relationship indices
        logger.info("Stage 3: Building relationship indices...")
        stage_start = time.time()
        if encoded_bag is not None:
            relationship_indices = self._build_relationship_indices_distributed(encoded_bag)
        else:
            relationship_indices = self._build_relationship_indices(encoded_annotations)
        stage_times["3. Relationship Indices"] = time.time() - stage_start

        # Stage 4: Write info file
        logger.info("Stage 4: Writing metadata...")
        stage_start = time.time()
        self.writer.write_info(
            coordinate_space=self.coordinate_space,
            annotation_type=self.annotation_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            properties=self.properties,
            relationships=self.relationships,
            spatial_levels=self.spatial_builder.levels,
            level_sharding_configs=self.level_sharding_configs if self.level_sharding_configs else None,
        )
        stage_times["4. Write Metadata"] = time.time() - stage_start

        # Stage 5: Write all indices
        logger.info("Stage 5: Writing indices...")
        stage_start = time.time()
        if encoded_bag is not None:
            # Keep data distributed - pass bag directly to avoid collection overhead
            # This is a critical performance optimization that avoids serialization
            # and network transfer of all annotations to the main process
            write_times = self._write_indices(None, encoded_bag, spatial_results, relationship_indices)
        else:
            # Non-distributed path: use collected annotations
            write_times = self._write_indices(encoded_annotations, None, spatial_results, relationship_indices)
        stage_times["5. Write Indices"] = time.time() - stage_start
        
        # Add detailed write breakdown
        write_total = sum(write_times.values())
        if write_total > 0:
            logger.info("  Write breakdown:")
            for write_type, write_time in write_times.items():
                if write_time > 0:
                    percentage = (write_time / write_total) * 100
                    logger.info(f"    {write_type}: {write_time:.2f}s ({percentage:.1f}% of write stage)")

        # Calculate total time and print profiling summary
        total_time = time.time() - pipeline_start
        
        logger.info("=" * 80)
        logger.info("PIPELINE PROFILING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Stage':<30} {'Time (s)':<12} {'% of Total':<12}")
        logger.info("-" * 80)
        
        for stage_name, stage_time in stage_times.items():
            percentage = (stage_time / total_time) * 100
            logger.info(f"{stage_name:<30} {stage_time:<12.2f} {percentage:<12.1f}%")
        
        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<30} {total_time:<12.2f} {'100.0%':<12}")
        logger.info("=" * 80)
        
        if total_annotations > 0:
            throughput = total_annotations / total_time
            logger.info(f"Overall throughput: {throughput:,.0f} annotations/second")
        
        logger.info("Pipeline complete!")

    def _encode_annotations(
        self, annotation_iterator: Iterator[tuple[Any, ...]]
    ) -> list[tuple[list[EncodedAnnotation], list[float], list[float]]]:
        """Encode annotations and compute bounding box.

        Returns:
            Tuple of (encoded_annotations, lower_bound, upper_bound)
        """
        encoded = []
        lower_bound = [float("inf")] * self.coordinate_space.rank
        upper_bound = [float("-inf")] * self.coordinate_space.rank

        for item in annotation_iterator:
            if len(item) < 2:
                raise ValueError(f"Annotation tuple must have at least (id, geometry), got {len(item)} items")

            ann_id = item[0]
            geometry = item[1]
            properties = item[2] if len(item) > 2 else {}
            relationships = item[3] if len(item) > 3 else []

            encoded_ann = self.encoder.encode_single(
                annotation_id=ann_id,
                geometry=geometry,
                properties=properties,
                relationships=relationships,
            )

            encoded.append(encoded_ann)

            # Update bounds
            for d, (min_val, max_val) in enumerate(encoded_ann.bounding_box):
                lower_bound[d] = min(lower_bound[d], min_val)
                upper_bound[d] = max(upper_bound[d], max_val)

        return encoded, lower_bound, upper_bound

    def _encode_annotations_distributed(
        self, annotations_bag: Any  # dask.bag.Bag
    ) -> tuple[Any, list[float], list[float], int]:
        """Encode annotations and compute bounding box using Dask.
        
        This processes annotations in distributed partitions without loading
        all annotations into memory on a single worker.
        
        Args:
            annotations_bag: Dask bag of (id, geometry, properties, relationships) tuples
            
        Returns:
            Tuple of (encoded_bag, lower_bound, upper_bound, total_annotations)
        """
        if db is None:
            raise ImportError("Dask is required for distributed processing. Install with: pip install dask")
        
        # Extract encoder config (serializable) instead of passing encoder object
        annotation_type = self.annotation_type
        rank = self.coordinate_space.rank
        
        # Convert properties to plain dicts to ensure full serializability
        # AnnotationPropertySpec objects might contain unpicklable attributes
        properties_dicts = []
        for prop in self.properties:
            prop_dict = {
                "id": prop.id,
                "type": prop.type,
                "description": getattr(prop, "description", None),
                "default": getattr(prop, "default", None),
                "enum_values": getattr(prop, "enum_values", None),
                "enum_labels": getattr(prop, "enum_labels", None),
            }
            properties_dicts.append(prop_dict)
        
        # Use a callable class that can be pickled (avoids lambda/partial issues)
        encode_func = _EncodePartitionCallable(
            annotation_type=annotation_type,
            rank=rank,
            properties=properties_dicts,
        )
        
        # Process partitions in parallel - encoding happens distributed across workers
        # Each partition returns a list containing a single _PartitionEncodingResult
        # map_partitions yields each element, so we get _PartitionEncodingResult objects directly
        encoded_results_bag = annotations_bag.map_partitions(encode_func)
        
        # Compute total annotations count (distributed)
        total_annotations = encoded_results_bag.map(_get_encoded_length).sum().compute()
        
        # Aggregate bounds using Dask fold (tree reduction)
        rank_value = rank  # Store in local variable for closure
        
        def reduce_bounds(acc, result):
            """Reduce function to compute global bounds.
            
            In Dask's fold, the second argument can be either:
            - A partition result (_PartitionEncodingResult) when processing partition data
            - An accumulator tuple (prev_lower, prev_upper, rank) when merging accumulators
            """
            # Extract bounds from result (could be _PartitionEncodingResult or accumulator tuple)
            if isinstance(result, _PartitionEncodingResult):
                result_lower = result.local_lower
                result_upper = result.local_upper
            elif isinstance(result, (list, tuple)) and len(result) == 3:
                # This is an accumulator tuple from a previous merge
                result_lower, result_upper, _ = result
            else:
                raise TypeError(
                    f"Unexpected result type in reduce_bounds: {type(result)}, "
                    f"expected _PartitionEncodingResult or (list, list, int) tuple"
                )
            
            if acc is None:
                return (result_lower[:], result_upper[:], rank_value)
            else:
                prev_lower, prev_upper, _ = acc
                new_lower = [min(prev_lower[d], result_lower[d]) for d in range(rank_value)]
                new_upper = [max(prev_upper[d], result_upper[d]) for d in range(rank_value)]
                return (new_lower, new_upper, rank_value)
        
        # Compute global bounds
        bounds_result = encoded_results_bag.fold(reduce_bounds, initial=None).compute()
        if bounds_result is None:
            raise ValueError("No annotations to process")
        lower_bound, upper_bound, _ = bounds_result
        
        # Extract encoded annotations into a bag (flatten all encoded lists)
        encoded_bag = encoded_results_bag.map(_get_encoded_list).flatten()
        
        # Persist the encoded bag so it can be reused across multiple stages
        # without recomputation. This is critical for performance since we use
        # the encoded_bag for spatial index, relationship indices, and by_id writing.
        if dd is None:
            raise ImportError("Dask is required for distributed processing. Install with: pip install dask")
        logger.info("Persisting encoded annotations to worker memory...")
        encoded_bag = encoded_bag.persist()
        # Wait for persistence to complete
        dd.wait(encoded_bag)
        logger.info(f"Encoded {total_annotations} annotations across distributed partitions (persisted in worker memory)")
        
        return encoded_bag, lower_bound, upper_bound, total_annotations

    def _build_spatial_index(
        self, encoded_annotations: list[EncodedAnnotation]
    ) -> dict[int, dict[tuple[int, ...], list[EncodedAnnotation]]]:
        """Build multi-level spatial index.

        Returns:
            Dictionary mapping level -> cell_coords -> emitted_annotations
        """
        if self.spatial_builder is None:
            raise RuntimeError("Spatial index builder not initialized")

        results = {}
        remaining_by_cell: dict[tuple[int, ...], list[EncodedAnnotation]] = {}

        # Initialize remaining annotations for level 0
        # All annotations start at the root cell (0, 0, ...)
        root_cell = tuple([0] * self.coordinate_space.rank)
        remaining_by_cell[root_cell] = encoded_annotations

        # Process levels dynamically - compute next level only if annotations remain
        # According to spec: "Continue generating successively finer spatial index levels until no annotations remain"
        level_num = 0
        while True:
            try:
                # Compute level on-demand
                level = self.spatial_builder._ensure_level(level_num)
            except (ValueError, IndexError):
                # Can't compute more levels (can't subdivide further)
                logger.warning(f"Cannot compute level {level_num}, stopping level generation")
                break
            
            logger.info(f"Processing level {level.level} (grid_shape={level.grid_shape}, chunk_size={level.chunk_size})...")
            emitted_by_cell, next_remaining = self.spatial_builder.build_level(
                level, remaining_by_cell, seed=42
            )
            results[level.level] = emitted_by_cell
            
            # Count emitted and remaining
            total_emitted = sum(len(anns) for anns in emitted_by_cell.values())
            total_remaining = sum(len(anns) for anns in next_remaining.values())
            logger.debug(f"Emitted {total_emitted} annotations, {total_remaining} remaining for next level")
            
            remaining_by_cell = next_remaining
            
            # Stop if no annotations remain (per spec: "until no annotations remain")
            if not remaining_by_cell or all(not anns for anns in remaining_by_cell.values()):
                logger.info(f"No remaining annotations after level {level.level}, stopping level generation")
                break
            
            level_num += 1

        return results

    def _build_spatial_index_distributed(
        self, encoded_bag: Any, total_annotations: int  # dask.bag.Bag
    ) -> dict[int, dict[tuple[int, ...], list[EncodedAnnotation]]]:
        """Build multi-level spatial index using Dask in a fully distributed fashion.
        
        This implements the distributed spatial index building approach:
        1. Estimate number of levels conservatively
        2. For each level:
           a. Map: Compute which cell each annotation belongs to (distributed)
           b. GroupBy: Group annotations by cell (distributed, lazy)
           c. Take: Take first N per cell (distributed)
           d. Collect emitted annotations and remaining annotations
           e. Propagate remaining to child cells for next level
        
        IMPORTANT: The input data (encoded_bag) must be pre-randomized to avoid spatial biases.
        If partitions have strong spatial biases (e.g., CSV rows in spatial order), the spatial
        index will be unbalanced and computation will not be well distributed. See documentation
        for data source adapters for guidance on ensuring proper randomization.
        
        Args:
            encoded_bag: Dask bag of EncodedAnnotation objects (must be pre-randomized)
            total_annotations: Total number of annotations (for level estimation)
            
        Returns:
            Dictionary mapping level -> cell_coords -> emitted_annotations
        """
        if self.spatial_builder is None:
            raise RuntimeError("Spatial index builder not initialized")
        
        if db is None:
            raise ImportError("Dask is required for distributed processing. Install with: pip install dask")
        
        # Estimate number of levels conservatively
        # We need at least enough cells to hold all annotations (total_annotations / limit)
        # Grid grows as 2^(level * rank) in worst case, so estimate:
        # 2^(max_level * rank) >= total_annotations / limit
        # max_level >= log2(total_annotations / limit) / rank
        min_cells_needed = max(1, math.ceil(total_annotations / self.spatial_limit))
        max_level_estimate = math.ceil(math.log2(min_cells_needed) / self.coordinate_space.rank) + 2  # +2 for safety
        logger.info(f"Estimated {max_level_estimate} spatial levels (based on {total_annotations} annotations, limit={self.spatial_limit})")
        
        results = {}
        
        # Use encoded_bag directly - randomization must be handled by data source
        # See documentation for requirements on pre-randomization to avoid spatial biases
        remaining_bag = encoded_bag  # Start with all annotations
        
        # Process levels hierarchically
        for level_num in range(max_level_estimate + 1):  # +1 to be safe
            try:
                level = self.spatial_builder._ensure_level(level_num)
            except (ValueError, IndexError):
                # Can't compute more levels
                logger.info(f"Cannot compute level {level_num}, stopping")
                break
            
            logger.info(f"Processing level {level.level} (grid_shape={level.grid_shape}, chunk_size={level.chunk_size})...")
            
            # Check how many annotations we're starting with (only if debug logging enabled)
            # This count() operation is expensive as it materializes the entire bag, so we skip it in normal operation
            if logger.isEnabledFor(logging.DEBUG):
                remaining_count_before = remaining_bag.count().compute()
                logger.debug(f"Level {level.level}: Starting with {remaining_count_before} annotations")
            else:
                logger.info(f"Level {level.level}: Processing spatial index level...")
            
            # Step 1: Map annotations to their spatial cells
            # Each annotation can intersect multiple cells, so we emit (cell_coords, ann) for each intersection
            map_to_cells_func = _MapToCellsCallable(
                rank=self.coordinate_space.rank,
                lower_bound=list(level.lower_bound),
                chunk_size=list(level.chunk_size),
                grid_shape=level.grid_shape,
            )
            cell_annotations = remaining_bag.map(map_to_cells_func).flatten()
            
            # Step 2: Group by cell locally within each partition and take first N per partition
            # This avoids the expensive global merge by taking limit/num_partitions per partition per cell
            # We'll group by cell when writing, but data volume is much smaller
            # Estimate number of partitions (for limit_per_partition calculation)
            # We can't know exact number without computing, so use a conservative estimate
            try:
                n_workers = len(self.dask_client.scheduler_info()['workers'])
                estimated_partitions = max(1, n_workers * 2)
            except (KeyError, AttributeError):
                estimated_partitions = 8
            
            # Calculate limit per partition (ceiling to ensure we get at least limit total)
            limit_per_partition = max(1, (self.spatial_limit + estimated_partitions - 1) // estimated_partitions)
            logger.info(f"Level {level.level}: Taking first {limit_per_partition} per partition per cell (total limit={self.spatial_limit})")
            
            local_group_func = _LocalGroupAndTakeNCallable(limit_per_partition=limit_per_partition)
            partition_results = cell_annotations.map_partitions(local_group_func)
            
            # Extract emitted and remaining from partition results
            # Each partition returns [{'emitted': {...}, 'remaining': {...}}]
            # But Dask might return the dict directly in some cases
            def extract_emitted(partition_result) -> list:
                """Extract emitted annotations as (cell_coords, ann) tuples."""
                # Handle both list and dict cases
                if isinstance(partition_result, list):
                    if not partition_result:
                        return []
                    result_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
                elif isinstance(partition_result, dict):
                    result_dict = partition_result
                else:
                    return []
                
                emitted_dict = result_dict.get('emitted', {})
                # Flatten to (cell_coords, ann) tuples
                items = []
                for cell_coords, anns in emitted_dict.items():
                    for ann in anns:
                        items.append((cell_coords, ann))
                return items
            
            def extract_remaining(partition_result) -> list:
                """Extract remaining annotations as (cell_coords, ann) tuples."""
                # Handle both list and dict cases
                if isinstance(partition_result, list):
                    if not partition_result:
                        return []
                    result_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
                elif isinstance(partition_result, dict):
                    result_dict = partition_result
                else:
                    return []
                
                remaining_dict = result_dict.get('remaining', {})
                # Flatten to (cell_coords, ann) tuples
                items = []
                for cell_coords, anns in remaining_dict.items():
                    for ann in anns:
                        items.append((cell_coords, ann))
                return items
            
            # Count for logging (need to extract for counting, but won't use the flattened version)
            # Count emitted annotations across partitions
            def count_emitted(partition_result) -> int:
                """Count emitted annotations in a partition result."""
                if isinstance(partition_result, list):
                    if not partition_result:
                        return 0
                    result_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
                elif isinstance(partition_result, dict):
                    result_dict = partition_result
                else:
                    return 0
                emitted_dict = result_dict.get('emitted', {})
                return sum(len(anns) for anns in emitted_dict.values())
            
            def count_remaining(partition_result) -> int:
                """Count remaining annotations in a partition result."""
                if isinstance(partition_result, list):
                    if not partition_result:
                        return 0
                    result_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
                elif isinstance(partition_result, dict):
                    result_dict = partition_result
                else:
                    return 0
                remaining_dict = result_dict.get('remaining', {})
                return sum(len(anns) for anns in remaining_dict.values())
            
            # Compute both counts together in a single .compute() call for efficiency
            # This avoids materializing partition_results twice
            emitted_count_delayed = partition_results.map(count_emitted).sum()
            remaining_count_delayed = partition_results.map(count_remaining).sum()
            total_emitted, total_remaining = dask.compute(emitted_count_delayed, remaining_count_delayed)
            logger.info(f"Level {level.level}: After per-partition limiting: {total_remaining} remaining")
            
            # Convert partition results to bag format WITHOUT collecting to main process
            # Extract emitted dicts from each partition and convert to bag of (cell_coords, annotations) tuples
            # This happens distributed on workers
            logger.info(f"Level {level.level}: Converting partition results to bag format (distributed)...")
            
            def partition_result_to_emitted_tuples(partition_result):
                """Convert partition result to list of (cell_coords, annotations) tuples - runs on worker."""
                # Handle both list and dict cases
                if isinstance(partition_result, list):
                    if not partition_result:
                        return []
                    result_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
                elif isinstance(partition_result, dict):
                    result_dict = partition_result
                else:
                    return []
                
                emitted_dict = result_dict.get('emitted', {})
                # Convert to list of (cell_coords, annotations) tuples
                return [(cell_coords, anns) for cell_coords, anns in emitted_dict.items()]
            
            # Convert partition results to bag - each partition extracts its emitted tuples
            # This happens on workers, not main process
            emitted_partition_bag = partition_results.map(partition_result_to_emitted_tuples).flatten()
            
            # Now we need to merge cells that appear in multiple partitions
            # Use groupby to merge - this is a shuffle, but it's necessary to combine cells from different partitions
            # However, we've already done local grouping per partition, so this groupby is merging pre-grouped data
            # which is much smaller than the original annotations
            logger.info(f"Level {level.level}: Grouping emitted cells across partitions (necessary shuffle for merging)...")
            # Group by cell_coords to merge annotations from different partitions
            # The key is the first element (cell_coords)
            # MEMORY OPTIMIZATION: Don't persist - let writing trigger computation incrementally
            emitted_grouped = emitted_partition_bag.groupby(lambda x: x[0])
            
            # Extract and combine annotations from groupby result
            # groupby returns (cell_coords, [(cell_coords, anns), (cell_coords, anns), ...])
            # We need to combine all annotations for each cell
            def combine_cell_group(group_item):
                """Combine annotations from groupby result - runs on worker."""
                cell_coords, items = group_item
                # items is a list of (cell_coords, annotations) tuples
                # All have the same cell_coords, so we combine all annotations
                all_annotations = []
                for _, anns in items:
                    if anns:  # Only add non-empty lists
                        all_annotations.extend(anns)
                return (cell_coords, all_annotations)
            
            emitted_bag = emitted_grouped.map(combine_cell_group)
            
            # Extract remaining for next level (still need to flatten for mapping to child cells)
            remaining_flat = partition_results.map(extract_remaining).flatten()
            
            # Prepare remaining annotations for next level
            # Only create the callable if we can compute the next level
            try:
                next_level = self.spatial_builder._ensure_level(level.level + 1)
                
                if total_remaining > 0:
                    # Map remaining annotations to child cells (distributed)
                    # remaining_flat already contains (parent_cell_coords, ann) tuples
                    logger.info(f"Level {level.level}: Mapping {total_remaining} remaining annotations to child cells (distributed)...")
                    map_remaining_func = _MapRemainingToChildCellsCallable(
                        rank=self.coordinate_space.rank,
                        next_lower_bound=list(next_level.lower_bound),
                        next_chunk_size=list(next_level.chunk_size),
                        next_grid_shape=next_level.grid_shape,
                    )
                    remaining_mapped = remaining_flat.map(map_remaining_func).flatten()
                else:
                    remaining_mapped = None
                    remaining_count = 0
            except (ValueError, IndexError):
                # Can't compute next level - no remaining annotations
                remaining_count = 0
                remaining_mapped = None
            
            # Compute level-specific sharding config if using adaptive sharding
            level_sharding_config = None
            spatial_sharding = (self.writer.sharding_config or {}).get("spatial")
            if spatial_sharding and "max_annotations_per_shard" in spatial_sharding:
                # Use total_emitted already computed from partition_results (much cheaper than materializing emitted_bag)
                # This avoids the expensive operation: emitted_bag.map(lambda x: len(x[1])).sum().compute()
                total_emitted_at_level = total_emitted
                logger.info(f"Level {level.level}: Using {total_emitted_at_level} emitted annotations for adaptive sharding (counted from partition_results, no materialization overhead)")
                
                # Compute level-specific sharding config
                from .writer import _compute_shard_config_from_max_annotations
                level_sharding_config = _compute_shard_config_from_max_annotations(
                    spatial_sharding,
                    total_emitted_at_level,
                    spatial_sharding["max_annotations_per_shard"],
                )
                self.level_sharding_configs[level.level] = level_sharding_config
                num_shards = 1 << level_sharding_config["shard_bits"]
                logger.info(f"Level {level.level}: Using {num_shards} shards (shard_bits={level_sharding_config['shard_bits']}) for {total_emitted_at_level} annotations")
            
            # Write emitted annotations directly from distributed bag (no collection needed!)
            # MEMORY OPTIMIZATION: Don't persist - let writing trigger computation incrementally
            # This allows Dask to release memory for written cells as they complete
            
            # The emitted_bag already contains (cell_coords, emitted_annotations) tuples
            # No need to groupby again - each cell is already grouped
            # Write directly without persisting first
            logger.info(f"Level {level.level}: Writing emitted annotations directly from distributed bag (no collection)...")
            write_start = time.time()
            
            # Write directly from bag (no collection, no persist!)
            # Dask will compute cells incrementally as needed for writing
            total_written = self.writer.write_annotations_spatial_distributed(
                emitted_bag,
                level=level,
                level_sharding_config=level_sharding_config,
            )
            
            write_time = time.time() - write_start
            logger.info(f"Level {level.level}: Distributed writing took {write_time:.2f}s (no collection overhead!)")
            
            # Track that this level was processed (for consistency, though not used in Dask path)
            results[level.level] = {}
            # Clean up to help Dask garbage collect
            del emitted_bag
            
            if total_remaining == 0 or remaining_mapped is None:
                logger.info(f"No remaining annotations after level {level.level}")
                break
            
            # Create new remaining_bag for next level
            # remaining_mapped contains (child_cell_coords, ann) tuples
            # Extract annotations for next level
            logger.debug(f"Level {level.level}: Creating remaining bag for next level (distributed)...")
            remaining_bag = remaining_mapped.map(_get_second_element)  # Extract ann from (cell, ann)
            
            # Persist remaining_bag to keep it in memory for next level
            # This prevents the graph from growing too large
            remaining_bag = remaining_bag.persist()
            
            logger.info(
                f"Level {level.level}: written {total_written}, remaining {total_remaining}, "
                f"writing took {write_time:.2f}s"
            )
        
        return results

    def _build_relationship_indices(
        self, encoded_annotations: list[EncodedAnnotation]
    ) -> dict[str, dict[int, list[EncodedAnnotation]]]:
        """Build relationship indices.

        Returns:
            Dictionary mapping relationship_name -> object_id -> annotations
        """
        if not self.relationships:
            return {}
        
        relationship_indices = {rel: {} for rel in self.relationships}
        
        for ann in encoded_annotations:
            # ann.relationships is a list of lists, where each index corresponds to a relationship type
            # self.relationships[i] corresponds to ann.relationships[i]
            if hasattr(ann, 'relationships') and ann.relationships:
                for rel_idx, rel_name in enumerate(self.relationships):
                    if rel_idx < len(ann.relationships) and ann.relationships[rel_idx]:
                        for obj_id in ann.relationships[rel_idx]:
                            if obj_id not in relationship_indices[rel_name]:
                                relationship_indices[rel_name][obj_id] = []
                            relationship_indices[rel_name][obj_id].append(ann)
        
        return relationship_indices

    def _build_relationship_indices_distributed(
        self, encoded_bag: Any  # dask.bag.Bag
    ) -> Any:  # dask.bag.Bag of ((relationship, object_id), annotations) tuples
        """Build relationship indices using Dask.
        
        Args:
            encoded_bag: Dask bag of EncodedAnnotation objects
            
        Returns:
            Dask bag of ((relationship_name, object_id), annotations) tuples
            (distributed, not collected to main process)
        """
        if not self.relationships:
            return {}
        
        if db is None:
            raise ImportError("Dask is required for distributed processing. Install with: pip install dask")
        
        # Build indices in parallel per partition, then aggregate
        # Extract relationships list to avoid capturing self
        relationships_list = list(self.relationships)
        
        def build_relationship_partition(partition):
            """Build relationship indices for a partition."""
            local_indices = {rel: {} for rel in relationships_list}
            
            for ann in partition:
                # ann.relationships is a list of lists, where each index corresponds to a relationship type
                # relationships_list[i] corresponds to ann.relationships[i]
                if hasattr(ann, 'relationships') and ann.relationships:
                    for rel_idx, rel_name in enumerate(relationships_list):
                        if rel_idx < len(ann.relationships) and ann.relationships[rel_idx]:
                            for obj_id in ann.relationships[rel_idx]:
                                if obj_id not in local_indices[rel_name]:
                                    local_indices[rel_name][obj_id] = []
                                local_indices[rel_name][obj_id].append(ann)
            
            # Wrap in list to prevent Dask from iterating over the dict (dicts are iterable over keys)
            return [local_indices]
        
        # Process partitions in parallel
        partition_indices = encoded_bag.map_partitions(build_relationship_partition)
        
        # OPTIMIZATION: Convert partition-level indices to a bag of ((relationship, object_id), [annotations]) tuples
        # Keep the groups from local partitioning instead of flattening to individual annotations.
        # This reduces network shuffle from 5M individual annotations to ~50k groups (100x reduction!)
        def convert_partition_indices_to_group_tuples(partition_result):
            """Convert partition-level indices dict to list of ((rel, obj_id), [anns]) tuples (keeping groups)."""
            result_list = []
            # Handle wrapped format
            if isinstance(partition_result, list):
                if not partition_result:
                    return []
                indices_dict = partition_result[0] if isinstance(partition_result[0], dict) else {}
            elif isinstance(partition_result, dict):
                indices_dict = partition_result
            else:
                return []
            
            # Keep groups intact - convert {rel: {obj_id: [anns]}} to [((rel, obj_id), [anns]), ...]
            for rel_name, obj_indices in indices_dict.items():
                for obj_id, anns in obj_indices.items():
                    if anns:  # Only include non-empty groups
                        result_list.append(((rel_name, obj_id), anns))
            return result_list
        
        # Convert partition results to bag of group tuples (keep groups, don't flatten to individual annotations!)
        relationship_groups_bag = partition_indices.map(convert_partition_indices_to_group_tuples).flatten()
        
        # Group by (relationship, object_id) to merge groups for the same key
        # OPTIMIZATION: We're now shuffling ~50k groups instead of 5M individual annotations (100x reduction in network traffic)
        # MEMORY OPTIMIZATION: Don't persist - let writing trigger computation incrementally
        logger.info("Grouping relationship indices by (relationship, object_id) (distributed shuffle of groups, not individual annotations)...")
        grouped = relationship_groups_bag.groupby(lambda x: x[0])  # Group by (rel_name, obj_id)
        
        # Merge annotations from groupby result: (key, [(key, [anns1]), (key, [anns2]), ...]) -> (key, [anns1 + anns2 + ...])
        def merge_annotation_groups(group_item):
            """Merge annotation lists from groupby result."""
            (rel_name, obj_id), items = group_item
            # items is a list of ((rel_name, obj_id), [anns]) tuples
            # Merge all annotation lists into one
            all_annotations = []
            for _, anns_list in items:
                all_annotations.extend(anns_list)
            return ((rel_name, obj_id), all_annotations)
        
        relationship_groups_bag = grouped.map(merge_annotation_groups)
        
        # Return the bag instead of collecting to dict
        # The caller (write_annotations_relationship_distributed) will handle writing from the bag
        return relationship_groups_bag

    def _write_indices(
        self,
        encoded_annotations: list[EncodedAnnotation] | None,
        encoded_bag: Any | None,  # dask.bag.Bag or None
        spatial_results: dict[int, dict[tuple[int, ...], list[EncodedAnnotation]]],
        relationship_indices: dict[str, dict[int, list[EncodedAnnotation]]] | Any,  # dict for non-distributed, bag for distributed
    ) -> dict[str, float]:
        """Write all indices to storage.
        
        Args:
            encoded_annotations: List of encoded annotations (for non-distributed path)
            encoded_bag: Dask bag of encoded annotations (for distributed path)
            spatial_results: Spatial index results
            relationship_indices: Relationship index results
            
        Returns:
            Dictionary mapping index type to time spent writing (for profiling)
        """
        write_times = {}
        
        # Write by_id index
        write_start = time.time()
        if self.use_dask and self.dask_client is not None and encoded_bag is not None:
            # Distributed path: use the persisted bag directly (no collection!)
            # This avoids massive serialization and network transfer overhead
            logger.info("Writing by_id index using distributed bag (no collection)...")
            self.writer.write_annotations_by_id_distributed(encoded_bag)
        else:
            # Non-distributed path: use the collected list
            if encoded_annotations is None:
                raise ValueError("encoded_annotations must be provided for non-distributed path")
            self.writer.write_annotations_by_id_batch(encoded_annotations)
        write_times["by_id"] = time.time() - write_start
        
        # Write spatial indices
        # Note: If using Dask, spatial indices are already written during _build_spatial_index_distributed
        # We only need to write them here if NOT using Dask
        write_start = time.time()
        if not (self.use_dask and self.dask_client is not None):
            for level_num, emitted_by_cell in spatial_results.items():
                level = self.spatial_builder.levels[level_num]
                for cell_coords, annotations in emitted_by_cell.items():
                    self.writer.write_annotations_spatial(
                        level=level,
                        cell_coords=cell_coords,
                        annotations=annotations,
                    )
            write_times["spatial"] = time.time() - write_start
            logger.info("Spatial indices written")
        else:
            # With Dask, spatial indices are already written, just log
            logger.info("Spatial indices already written via distributed processing")
            write_times["spatial"] = 0.0  # Already written during spatial index building
        
        # Write relationship indices
        write_start = time.time()
        if self.use_dask and self.dask_client is not None:
            # Distributed path: relationship_indices is a bag, write directly (no collection!)
            logger.info("Writing relationship indices using distributed processing (no collection)...")
            self.writer.write_annotations_relationship_distributed(relationship_indices)
        else:
            # Non-distributed path: relationship_indices is a dict, write from main process
            for rel_name, obj_indices in relationship_indices.items():
                for obj_id, annotations in obj_indices.items():
                    self.writer.write_annotations_relationship(
                        relationship=rel_name,
                        object_id=obj_id,
                        annotations=annotations,
                    )
        write_times["relationships"] = time.time() - write_start
        
        return write_times
