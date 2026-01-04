"""Annotation encoding and decoding utilities for the precomputed format."""

import struct
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

# Import constants from parent precomputed_python module
from .. import _PROPERTY_DTYPES, AnnotationType


class EncodedAnnotation(NamedTuple):
    """A single encoded annotation with metadata."""

    id: int
    """Unique uint64 annotation id."""
    encoded: bytes
    """Binary encoded annotation data (geometry + properties, no relationships)."""
    relationships: Sequence[Sequence[int]]
    """List of relationship lists, one per relationship type."""
    bounding_box: tuple[tuple[float, float], ...]
    """Axis-aligned bounding box as ((min_x, max_x), (min_y, max_y), ...)."""


class AnnotationEncoder:
    """Encodes and decodes annotations in the precomputed format."""

    def __init__(
        self,
        annotation_type: AnnotationType,
        rank: int,
        properties: Sequence[Any] = (),
    ):
        """Initialize encoder.

        Args:
            annotation_type: Type of annotation geometry (point, line, etc.)
            rank: Number of spatial dimensions
            properties: List of property specifications
        """
        self.annotation_type = annotation_type
        self.rank = rank
        self.properties = list(properties)
        self.properties_sorted = sorted(
            self.properties, key=lambda p: -_PROPERTY_DTYPES[p.type][1]
        )
        self._dtype = self._get_dtype_for_geometry(
            annotation_type, rank
        ) + self._get_dtype_for_properties(self.properties_sorted)

    def _get_dtype_for_geometry(self, annotation_type: AnnotationType, rank: int):
        """Get numpy dtype for geometry portion."""
        if annotation_type == "point":
            geometry_size = rank
        elif annotation_type == "polyline":
            return [("num_points", "<u4")]
        else:
            geometry_size = 2 * rank
        return [("geometry", "<f4", geometry_size)]

    def _get_dtype_for_properties(self, properties: Sequence[Any]):
        """Get numpy dtype for properties portion with proper alignment."""
        dtype = []
        offset = 0
        for i, p in enumerate(properties):
            dtype_entry, alignment = _PROPERTY_DTYPES[p.type]
            if offset % alignment:
                padded_offset = (offset + alignment - 1) // alignment * alignment
                padding = padded_offset - offset
                dtype.append((f"padding{offset}", "|u1", (padding,)))
                offset += padding
            dtype.append((f"property{i}", *dtype_entry))
            size = np.dtype(dtype[-1:]).itemsize
            offset += size
        # Final padding to 4-byte alignment
        alignment = 4
        if offset % alignment:
            padded_offset = (offset + alignment - 1) // alignment * alignment
            padding = padded_offset - offset
            dtype.append((f"padding{offset}", "|u1", (padding,)))
        return dtype

    def encode_single(
        self,
        annotation_id: int,
        geometry: Sequence[float],
        properties: dict[str, Any] | None = None,
        relationships: Sequence[Sequence[int]] | None = None,
    ) -> EncodedAnnotation:
        """Encode a single annotation.

        Args:
            annotation_id: Unique uint64 id
            geometry: Geometry data (position, endpoints, etc.)
            properties: Dictionary of property values
            relationships: List of relationship lists

        Returns:
            EncodedAnnotation with encoded bytes and metadata
        """
        if properties is None:
            properties = {}
        if relationships is None:
            relationships = []

        # Handle polyline specially
        if self.annotation_type == "polyline":
            num_points = len(geometry) // self.rank
            # For polyline: num_points (uint32) + geometry (float32 * num_points * rank) + properties
            dtype = [("num_points", "<u4")]
            encoded = np.zeros(shape=(), dtype=dtype)
            encoded[()]["num_points"] = num_points
            geometry_array = np.array(geometry, dtype=np.float32)
            geometry_bytes = geometry_array.tobytes()
            # Properties will be appended after geometry
            properties_dtype = self._get_dtype_for_properties(self.properties_sorted)
            if properties_dtype:
                properties_array = np.zeros(shape=(), dtype=properties_dtype)
                encoded_bytes = encoded.tobytes() + geometry_bytes + properties_array.tobytes()
            else:
                encoded_bytes = encoded.tobytes() + geometry_bytes
        else:
            encoded = np.zeros(shape=(), dtype=self._dtype)
            encoded[()]["geometry"] = geometry
            encoded_bytes = None  # Will be set below

        # Add property values
        if self.annotation_type == "polyline":
            # For polyline, properties come after geometry
            properties_dtype = self._get_dtype_for_properties(self.properties_sorted)
            if properties_dtype:
                properties_array = np.zeros(shape=(), dtype=properties_dtype)
                for i, p in enumerate(self.properties_sorted):
                    value = properties.get(p.id, p.default)
                    if value is not None:
                        if isinstance(value, str) and p.type in ("rgb", "rgba"):
                            if p.type == "rgb":
                                value = self._convert_rgb_to_uint8(value)
                            else:
                                value = self._convert_rgba_to_uint8(value)
                        properties_array[()][f"property{i}"] = value
                encoded_bytes = encoded.tobytes() + geometry_bytes + properties_array.tobytes()
            else:
                encoded_bytes = encoded.tobytes() + geometry_bytes
        else:
            for i, p in enumerate(self.properties_sorted):
                value = properties.get(p.id, p.default)
                if value is not None:
                    if isinstance(value, str) and p.type in ("rgb", "rgba"):
                        if p.type == "rgb":
                            value = self._convert_rgb_to_uint8(value)
                        else:
                            value = self._convert_rgba_to_uint8(value)
                    encoded[()][f"property{i}"] = value
            encoded_bytes = encoded.tobytes()

        # Compute bounding box
        bounding_box = self._compute_bounding_box(geometry)

        return EncodedAnnotation(
            id=annotation_id,
            encoded=encoded_bytes,
            relationships=list(relationships),
            bounding_box=bounding_box,
        )

    def _compute_bounding_box(self, geometry: Sequence[float]) -> tuple[tuple[float, float], ...]:
        """Compute axis-aligned bounding box for geometry."""
        if self.annotation_type == "point":
            pos = np.array(geometry)
            return tuple((float(p), float(p)) for p in pos)
        elif self.annotation_type == "polyline":
            points = np.array(geometry).reshape(-1, self.rank)
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            return tuple((float(mi), float(ma)) for mi, ma in zip(mins, maxs))
        else:
            # Two points: point_a and point_b
            coords = np.array(geometry).reshape(2, self.rank)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            return tuple((float(mi), float(ma)) for mi, ma in zip(mins, maxs))

    def encode_multiple(self, annotations: Sequence[EncodedAnnotation]) -> bytes:
        """Encode multiple annotations in the format used by spatial and relationship indices.

        Format:
        - uint64le: count
        - For each annotation: encoded bytes (geometry + properties)
        - For each annotation: uint64le: annotation id

        Args:
            annotations: Sequence of encoded annotations

        Returns:
            Binary encoded data
        """
        buffer = bytearray()
        buffer.extend(struct.pack("<Q", len(annotations)))
        for ann in annotations:
            buffer.extend(ann.encoded)
        for ann in annotations:
            buffer.extend(struct.pack("<Q", ann.id))
        return bytes(buffer)

    def encode_single_with_relationships(self, annotation: EncodedAnnotation) -> bytes:
        """Encode a single annotation with relationships (for by_id index).

        Format:
        - encoded bytes (geometry + properties)
        - For each relationship:
          - uint32le: count of related ids
          - For each id: uint64le: related id

        Args:
            annotation: Encoded annotation

        Returns:
            Binary encoded data
        """
        buffer = bytearray(annotation.encoded)
        for related_ids in annotation.relationships:
            buffer.extend(struct.pack("<I", len(related_ids)))
            for related_id in related_ids:
                buffer.extend(struct.pack("<Q", related_id))
        return bytes(buffer)

    @staticmethod
    def _convert_rgb_to_uint8(rgb: str) -> tuple[int, int, int]:
        """Convert RGB hex string to uint8 tuple."""
        if rgb.startswith("#"):
            rgb = rgb[1:]
        if len(rgb) != 6:
            raise ValueError(f"Invalid RGB format: {rgb}")
        return (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))

    @staticmethod
    def _convert_rgba_to_uint8(rgba: str) -> tuple[int, int, int, int]:
        """Convert RGBA hex string to uint8 tuple."""
        if rgba.startswith("#"):
            rgba = rgba[1:]
        if len(rgba) != 8:
            raise ValueError(f"Invalid RGBA format: {rgba}")
        color = AnnotationEncoder._convert_rgb_to_uint8(rgba[:6])
        alpha = int(rgba[6:8], 16)
        return (*color, alpha)

