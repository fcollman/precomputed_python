"""Spatial index computation for precomputed annotations.

This module implements the multi-level spatial index as specified in the
precomputed annotation format. The key challenge is handling the global
constraint that maxCount(level) must be computed across all cells before
sampling can occur.
"""

import logging
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from .encoder import EncodedAnnotation

logger = logging.getLogger(__name__)


class SpatialCell:
    """Represents a spatial grid cell with its annotations."""

    def __init__(self, cell_coords: tuple[int, ...], annotations: list[EncodedAnnotation]):
        """Initialize spatial cell.

        Args:
            cell_coords: Grid cell coordinates
            annotations: List of annotations intersecting this cell
        """
        self.cell_coords = cell_coords
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)


class SpatialIndexLevel:
    """Represents a single level of the spatial index."""

    def __init__(
        self,
        level: int,
        grid_shape: tuple[int, ...],
        chunk_size: tuple[float, ...],
        lower_bound: tuple[float, ...],
    ):
        """Initialize spatial index level.

        Args:
            level: Level number (0 = coarsest)
            grid_shape: Number of cells along each dimension
            chunk_size: Size of each cell in physical units
            lower_bound: Lower bound of coordinate space
        """
        self.level = level
        self.grid_shape = grid_shape
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.rank = len(grid_shape)

    def get_cell_bounds(self, cell_coords: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
        """Get physical bounds for a grid cell.

        Returns:
            Tuple of (min, max) bounds for each dimension
        """
        bounds = []
        for d in range(self.rank):
            min_val = self.lower_bound[d] + cell_coords[d] * self.chunk_size[d]
            max_val = min_val + self.chunk_size[d]
            bounds.append((min_val, max_val))
        return tuple(bounds)

    def get_child_cells(self, parent_cell: tuple[int, ...], child_level: "SpatialIndexLevel") -> list[tuple[int, ...]]:
        """Get child cells in the next level that are contained within a parent cell.

        Args:
            parent_cell: Parent cell coordinates
            child_level: Next level spatial index

        Returns:
            List of child cell coordinates
        """
        parent_bounds = self.get_cell_bounds(parent_cell)
        child_cells = []

        # Compute how many child cells fit in each dimension
        cells_per_dim = []
        for d in range(self.rank):
            parent_size = self.chunk_size[d]
            child_size = child_level.chunk_size[d]
            if parent_size % child_size != 0:
                raise ValueError(
                    f"Child chunk_size must evenly divide parent: "
                    f"{parent_size} / {child_size} in dimension {d}"
                )
            cells_per_dim.append(int(parent_size / child_size))

        # Generate all child cell coordinates
        def generate_children(coords: list[int], dim: int):
            if dim == self.rank:
                # Check if this child cell is within parent bounds
                child_bounds = child_level.get_cell_bounds(tuple(coords))
                # Simple check: child min must be >= parent min, child max <= parent max
                # (More precise intersection check could be added)
                child_cells.append(tuple(coords))
                return
            for i in range(cells_per_dim[dim]):
                child_coord = parent_cell[dim] * cells_per_dim[dim] + i
                generate_children(coords + [child_coord], dim + 1)

        generate_children([], 0)
        return child_cells

    def cell_coords_to_key(self, cell_coords: tuple[int, ...]) -> str:
        """Convert cell coordinates to file key (unsharded format).

        Args:
            cell_coords: Grid cell coordinates

        Returns:
            String key (e.g., "1_2_3")
        """
        return "_".join(str(c) for c in cell_coords)


def intersects_cell(annotation: EncodedAnnotation, cell_bounds: tuple[tuple[float, float], ...]) -> bool:
    """Check if an annotation intersects a spatial cell.

    Args:
        annotation: Encoded annotation
        cell_bounds: Cell bounds as ((min_x, max_x), (min_y, max_y), ...)

    Returns:
        True if annotation intersects the cell
    """
    ann_bbox = annotation.bounding_box

    # Check if annotation bounding box overlaps with cell bounds
    for d in range(len(cell_bounds)):
        ann_min, ann_max = ann_bbox[d]
        cell_min, cell_max = cell_bounds[d]

        # No overlap if annotation is completely before or after cell
        if ann_max < cell_min or ann_min > cell_max:
            return False

    return True


class SpatialIndexBuilder:
    """Builds multi-level spatial index according to the specification.

    The spatial index is computed level by level:
    1. For each cell, compute remaining_annotations(level, cell)
    2. Compute maxCount(level) = max over all cells
    3. For each cell, sample annotations with probability min(1, limit / maxCount)
    4. Propagate remaining annotations to child cells
    """

    def __init__(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        initial_chunk_size: Sequence[float],
        limit: int,
        rank: int,
    ):
        """Initialize spatial index builder.

        Args:
            lower_bound: Lower bound of coordinate space
            upper_bound: Upper bound of coordinate space
            initial_chunk_size: Chunk size for coarsest level
            limit: Maximum annotations per cell
            rank: Number of spatial dimensions
        """
        self.lower_bound = tuple(lower_bound)
        self.upper_bound = tuple(upper_bound)
        self.rank = rank
        self.limit = limit

        # Compute levels lazily (on-demand) instead of all upfront
        # This avoids computing unnecessary levels when annotations are exhausted early
        self.levels: list[SpatialIndexLevel] = []
        self._last_chunk_size: list[float] | None = None
        
        # Initialize level 0
        self._ensure_level(0)

    def _ensure_level(self, level_num: int) -> SpatialIndexLevel:
        """Ensure a level exists, computing it if necessary.
        
        Args:
            level_num: The level number to ensure exists
            
        Returns:
            The SpatialIndexLevel for the requested level
        """
        # If we already have this level, return it
        if level_num < len(self.levels):
            return self.levels[level_num]
        
        # Compute levels up to the requested one
        if level_num == 0:
            # Level 0: grid_shape = (1,1,1), chunk_size = upper_bound - lower_bound
            level_0_chunk_size = [
                self.upper_bound[d] - self.lower_bound[d] 
                for d in range(self.rank)
            ]
            self._last_chunk_size = level_0_chunk_size
            grid_shape = [1] * self.rank
            
            level = SpatialIndexLevel(
                level=0,
                grid_shape=tuple(grid_shape),
                chunk_size=tuple(level_0_chunk_size),
                lower_bound=self.lower_bound,
            )
            self.levels.append(level)
            logger.debug(f"Computed level 0: grid_shape={grid_shape}, chunk_size={[round(c, 2) for c in level_0_chunk_size]}")
            return level
        
        # Compute subsequent levels
        # Start from the last computed level
        if self._last_chunk_size is None:
            # Shouldn't happen if level 0 was computed, but handle it
            self._ensure_level(0)
        
        current_chunk_size = list(self._last_chunk_size)
        current_level = len(self.levels) - 1
        
        # Compute levels until we reach the requested one
        while len(self.levels) <= level_num:
            # Compute next level chunk_size
            next_chunk_size = self._compute_next_chunk_size(current_chunk_size, current_level)
            
            # Check if we can make progress
            if next_chunk_size == current_chunk_size:
                # Can't subdivide further - return None to signal we're done
                # But first, check if we have the requested level
                if level_num < len(self.levels):
                    return self.levels[level_num]
                # We can't compute the requested level
                raise ValueError(f"Cannot compute level {level_num}: cannot subdivide further at level {current_level}")
            
            # Safety check for floating point precision
            if min(next_chunk_size) < 1e-10:
                raise ValueError(f"Cannot compute level {level_num}: chunk_size too small at level {current_level}")
            
            current_chunk_size = next_chunk_size
            current_level += 1
            
            # Compute grid_shape
            grid_shape = []
            for d in range(self.rank):
                size = self.upper_bound[d] - self.lower_bound[d]
                cells = int(size / current_chunk_size[d])
                # Verify it divides evenly (with small floating point tolerance)
                expected_size = cells * current_chunk_size[d]
                if abs(size - expected_size) > 1e-6:
                    # Adjust chunk_size to exactly divide the space
                    current_chunk_size[d] = size / cells
                grid_shape.append(cells)
            
            level = SpatialIndexLevel(
                level=current_level,
                grid_shape=tuple(grid_shape),
                chunk_size=tuple(current_chunk_size),
                lower_bound=self.lower_bound,
            )
            self.levels.append(level)
            self._last_chunk_size = current_chunk_size
            
            if current_level <= 5:  # Only print first few to reduce spam
                logger.debug(f"Computed level {current_level}: grid_shape={grid_shape}, chunk_size={[round(c, 2) for c in current_chunk_size]}")
        
        return self.levels[level_num]
    
    def _compute_next_chunk_size(self, current_chunk_size: list[float], current_level: int) -> list[float]:
        """Compute the chunk_size for the next level.
        
        Args:
            current_chunk_size: Current level's chunk_size
            current_level: Current level number
            
        Returns:
            Next level's chunk_size (or same if can't subdivide)
        """
        # Get current grid_shape to check if we can halve
        current_grid_shape = []
        for d in range(self.rank):
            size = self.upper_bound[d] - self.lower_bound[d]
            cells = int(size / current_chunk_size[d])
            current_grid_shape.append(cells)
        
        # First, try halving all dimensions
        halve_all = [current_chunk_size[d] / 2.0 for d in range(self.rank)]
        
        # Check if halving all is valid (creates more cells in all dimensions)
        can_halve_all = True
        for d in range(self.rank):
            size = self.upper_bound[d] - self.lower_bound[d]
            cells_if_halved = int(size / halve_all[d])
            if cells_if_halved <= current_grid_shape[d]:
                can_halve_all = False
                break
        
        if can_halve_all:
            # Compute isotropy for both options
            current_ratio = max(current_chunk_size) / min(current_chunk_size) if min(current_chunk_size) > 0 else float('inf')
            halved_ratio = max(halve_all) / min(halve_all) if min(halve_all) > 0 else float('inf')
            
            # Halve all if it makes cells more isotropic (smaller or equal ratio)
            if halved_ratio <= current_ratio:
                return halve_all
            else:
                # Halving all makes it less isotropic, so keep all same
                return list(current_chunk_size)
        else:
            # Can't halve all dimensions, try halving each dimension independently
            next_chunk_size = []
            for d in range(self.rank):
                half_size = current_chunk_size[d] / 2.0
                size = self.upper_bound[d] - self.lower_bound[d]
                cells_if_halved = int(size / half_size)
                
                # Can only halve if it creates more cells
                if cells_if_halved <= current_grid_shape[d]:
                    # Can't halve this dimension, keep same
                    next_chunk_size.append(current_chunk_size[d])
                    continue
                
                # Compute isotropy if we halve this dimension
                test_chunk_size = list(current_chunk_size)
                test_chunk_size[d] = half_size
                
                current_ratio = max(current_chunk_size) / min(current_chunk_size) if min(current_chunk_size) > 0 else float('inf')
                test_ratio = max(test_chunk_size) / min(test_chunk_size) if min(test_chunk_size) > 0 else float('inf')
                
                # Halve if it makes cells more isotropic (smaller or equal ratio)
                if test_ratio <= current_ratio:
                    next_chunk_size.append(half_size)
                else:
                    next_chunk_size.append(current_chunk_size[d])
            
            return next_chunk_size

    def _compute_levels(self, initial_chunk_size: Sequence[float]) -> list[SpatialIndexLevel]:
        """Compute spatial index levels.

        According to the spec:
        - Level 0 should have grid_shape=(1,1,1) with chunk_size = upper_bound - lower_bound
        - Each subsequent level should halve chunk_size (or keep equal) to make it more spatially isotropic
        - Each level's chunk_size must evenly divide the previous level's chunk_size

        Args:
            initial_chunk_size: Ignored - level 0 uses full space size. This parameter is kept
                                for API compatibility but not used.

        Returns:
            List of SpatialIndexLevel objects
        """
        levels = []
        
        # Level 0: grid_shape = (1,1,1), chunk_size = upper_bound - lower_bound
        level_0_chunk_size = [
            self.upper_bound[d] - self.lower_bound[d] 
            for d in range(self.rank)
        ]
        current_chunk_size = level_0_chunk_size
        level = 0

        while True:
            # Compute grid_shape such that grid_shape * chunk_size = upper_bound - lower_bound
            grid_shape = []
            for d in range(self.rank):
                size = self.upper_bound[d] - self.lower_bound[d]
                cells = int(size / current_chunk_size[d])
                # Verify it divides evenly (with small floating point tolerance)
                expected_size = cells * current_chunk_size[d]
                if abs(size - expected_size) > 1e-6:
                    # Adjust chunk_size to exactly divide the space
                    current_chunk_size[d] = size / cells
                grid_shape.append(cells)
            
            # Only add level if chunk_size is reasonable (not zero or negative)
            if min(current_chunk_size) <= 0:
                logger.debug(f"Stopped at level {level}: chunk_size became non-positive")
                break
            
            # Debug output (only for first few levels to reduce spam)
            if level <= 5:
                logger.debug(f"Computed level {level}: grid_shape={grid_shape}, chunk_size={[round(c, 2) for c in current_chunk_size]}")

            levels.append(
                SpatialIndexLevel(
                    level=level,
                    grid_shape=tuple(grid_shape),
                    chunk_size=tuple(current_chunk_size),
                    lower_bound=self.lower_bound,
                )
            )

            # Compute next level chunk_size (typically half in each dimension, or equal)
            # Note: We don't stop at single-cell resolution - we continue until
            # we can't halve any dimension further

            # Compute next chunk_size (more spatially isotropic)
            # According to spec: each component should be either equal to, or half of,
            # the corresponding component of the prior level chunk_size, whichever results
            # in a more spatially isotropic chunk.
            # 
            # Strategy: Try halving all dimensions, and compare the isotropy.
            # If halving all makes it more isotropic (or same), do it.
            # Otherwise, try halving only some dimensions.
            
            # First, try halving all dimensions
            halve_all = [current_chunk_size[d] / 2.0 for d in range(self.rank)]
            
            # Check if halving all is valid (creates more cells in all dimensions)
            can_halve_all = True
            for d in range(self.rank):
                size = self.upper_bound[d] - self.lower_bound[d]
                cells_if_halved = int(size / halve_all[d])
                if cells_if_halved <= grid_shape[d]:
                    can_halve_all = False
                    break
            
            if can_halve_all:
                # Compute isotropy for both options
                current_ratio = max(current_chunk_size) / min(current_chunk_size) if min(current_chunk_size) > 0 else float('inf')
                halved_ratio = max(halve_all) / min(halve_all) if min(halve_all) > 0 else float('inf')
                
                # Halve all if it makes cells more isotropic (smaller or equal ratio)
                if halved_ratio <= current_ratio:
                    next_chunk_size = halve_all
                else:
                    # Halving all makes it less isotropic, so keep all same
                    next_chunk_size = list(current_chunk_size)
            else:
                # Can't halve all dimensions, try halving each dimension independently
                next_chunk_size = []
                for d in range(self.rank):
                    half_size = current_chunk_size[d] / 2.0
                    size = self.upper_bound[d] - self.lower_bound[d]
                    cells_if_halved = int(size / half_size)
                    
                    # Can only halve if it creates more cells
                    if cells_if_halved <= grid_shape[d]:
                        # Can't halve this dimension, keep same
                        next_chunk_size.append(current_chunk_size[d])
                        continue
                    
                    # Compute isotropy if we halve this dimension
                    test_chunk_size = list(current_chunk_size)
                    test_chunk_size[d] = half_size
                    
                    current_ratio = max(current_chunk_size) / min(current_chunk_size) if min(current_chunk_size) > 0 else float('inf')
                    test_ratio = max(test_chunk_size) / min(test_chunk_size) if min(test_chunk_size) > 0 else float('inf')
                    
                    # Halve if it makes cells more isotropic (smaller or equal ratio)
                    if test_ratio <= current_ratio:
                        next_chunk_size.append(half_size)
                    else:
                        next_chunk_size.append(current_chunk_size[d])
            
            # Check if we made any progress
            if next_chunk_size == current_chunk_size:
                # No progress in any dimension, stop
                # According to spec: "Continue generating successively finer spatial index levels until no annotations remain"
                # But if we can't subdivide further, we stop here
                logger.debug(f"Stopped at level {level}: no progress (cannot subdivide further)")
                break
            
            current_chunk_size = next_chunk_size
            level += 1

            # Safety limit to prevent infinite loops (very high limit since we want to continue until annotations are gone)
            # But also check for floating point issues
            if level > 100 or min(current_chunk_size) < 1e-10:
                if min(current_chunk_size) < 1e-10:
                    logger.debug(f"Stopped at level {level}: chunk_size too small (floating point precision limit)")
                else:
                    logger.debug(f"Stopped at level {level}: safety limit reached")
                break

        return levels

    def compute_intersections(
        self,
        level: SpatialIndexLevel,
        annotations: Sequence[EncodedAnnotation],
    ) -> dict[tuple[int, ...], list[EncodedAnnotation]]:
        """Compute which annotations intersect each cell at a given level.

        Args:
            level: Spatial index level
            annotations: Annotations to process

        Returns:
            Dictionary mapping cell coordinates to list of intersecting annotations
        """
        cell_annotations: dict[tuple[int, ...], list[EncodedAnnotation]] = {}

        for ann in annotations:
            # Find all cells this annotation intersects
            ann_bbox = ann.bounding_box

            # Compute cell range for annotation
            cell_ranges = []
            for d in range(self.rank):
                ann_min, ann_max = ann_bbox[d]
                # Convert to cell coordinates
                cell_min = int((ann_min - level.lower_bound[d]) / level.chunk_size[d])
                cell_max = int((ann_max - level.lower_bound[d]) / level.chunk_size[d])
                # Clamp to valid range
                cell_min = max(0, cell_min)
                cell_max = min(level.grid_shape[d] - 1, cell_max)
                cell_ranges.append((cell_min, cell_max))

            # Generate all cells in the range
            def generate_cells(coords: list[int], dim: int):
                if dim == self.rank:
                    cell_coords = tuple(coords)
                    cell_bounds = level.get_cell_bounds(cell_coords)
                    if intersects_cell(ann, cell_bounds):
                        if cell_coords not in cell_annotations:
                            cell_annotations[cell_coords] = []
                        cell_annotations[cell_coords].append(ann)
                    return
                for c in range(cell_ranges[dim][0], cell_ranges[dim][1] + 1):
                    generate_cells(coords + [c], dim + 1)

            generate_cells([], 0)

        return cell_annotations

    def compute_max_count(
        self,
        cell_annotations: dict[tuple[int, ...], list[EncodedAnnotation]],
    ) -> int:
        """Compute maxCount for a level (maximum annotations in any cell).

        Args:
            cell_annotations: Dictionary mapping cells to annotations

        Returns:
            Maximum count across all cells
        """
        if not cell_annotations:
            return 0
        return max(len(anns) for anns in cell_annotations.values())

    def sample_annotations(
        self,
        annotations: list[EncodedAnnotation],
        max_count: int,
        limit: int,
        seed: int | None = None,
    ) -> tuple[list[EncodedAnnotation], list[EncodedAnnotation]]:
        """Sample annotations according to the specification.

        Each annotation is chosen with probability min(1, limit / maxCount).

        Args:
            annotations: Annotations to sample from
            max_count: Maximum count across all cells at this level
            limit: Maximum annotations per cell
            seed: Random seed for reproducibility

        Returns:
            Tuple of (emitted_annotations, remaining_annotations)
        """
        if seed is not None:
            random.seed(seed)

        if max_count == 0:
            return [], []

        probability = min(1.0, limit / max_count)
        emitted = []
        remaining = []

        for ann in annotations:
            if random.random() < probability:
                emitted.append(ann)
            else:
                remaining.append(ann)

        return emitted, remaining

    def build_level(
        self,
        level: SpatialIndexLevel,
        remaining_annotations: dict[tuple[int, ...], list[EncodedAnnotation]],
        seed: int | None = None,
    ) -> tuple[dict[tuple[int, ...], list[EncodedAnnotation]], dict[tuple[int, ...], list[EncodedAnnotation]]]:
        """Build a single level of the spatial index.

        Args:
            level: Spatial index level to build
            remaining_annotations: Annotations remaining from previous level (by cell)
            seed: Random seed for sampling

        Returns:
            Tuple of (emitted_annotations_by_cell, next_level_remaining_by_cell)
        """
        # Flatten remaining annotations to compute intersections
        all_remaining = []
        for anns in remaining_annotations.values():
            all_remaining.extend(anns)

        # Compute intersections for this level
        cell_annotations = self.compute_intersections(level, all_remaining)

        # Compute maxCount
        max_count = self.compute_max_count(cell_annotations)

        # Sample and emit annotations
        emitted_by_cell: dict[tuple[int, ...], list[EncodedAnnotation]] = {}
        next_level_remaining: dict[tuple[int, ...], list[EncodedAnnotation]] = {}

        for cell_coords, anns in cell_annotations.items():
            emitted, remaining = self.sample_annotations(anns, max_count, self.limit, seed)
            emitted_by_cell[cell_coords] = emitted

            # Propagate remaining to child cells
            # We always propagate remaining annotations to the next level (which will be computed on-demand)
            # The pipeline will stop when no annotations remain
            try:
                # Try to get the next level (compute it if needed)
                next_level = self._ensure_level(level.level + 1)
                child_cells = level.get_child_cells(cell_coords, next_level)
                for child_cell in child_cells:
                    if child_cell not in next_level_remaining:
                        next_level_remaining[child_cell] = []
                    # Check which remaining annotations intersect child cell
                    child_bounds = next_level.get_cell_bounds(child_cell)
                    for ann in remaining:
                        if intersects_cell(ann, child_bounds):
                            next_level_remaining[child_cell].append(ann)
            except (ValueError, IndexError):
                # Can't compute next level (can't subdivide further)
                # Remaining annotations are discarded (we've reached the finest level)
                pass

        return emitted_by_cell, next_level_remaining

