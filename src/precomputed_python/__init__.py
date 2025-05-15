"""Writes annotations in the Precomputed annotation format.

This provides a simple way to write annotations in the precomputed format, but
has a number of limitations that makes it suitable only for writing
up to a few million of annotations, and not beyond that.

- All annotations are buffered in memory.

- Only a single spatial index  of a fixed grid size is generated.
  No downsampling is performed. Consequently, Neuroglancer will be forced
  to download all annotations to render them in 3 dimensions.

"""

import asyncio
import json
import logging
import math
import numbers
import os
import pathlib
import struct
from collections import defaultdict
from collections.abc import Sequence
from itertools import product
from typing import Literal, NamedTuple, Optional, Union, cast

import numpy as np
import pandas as pd
import rtree  # type: ignore
import yaml
from jsonschema import RefResolver, ValidationError, validate
from neuroglancer.coordinate_space import CoordinateSpace

__version__ = "0.0.1"

try:
    import tensorstore as ts
except ImportError:
    logging.warning(
        "Sharded write support requires tensorstore."
        "Install with pip install tensorstore"
    )
    ts = None

import yaml
from jsonschema import RefResolver, ValidationError, validate
from neuroglancer import coordinate_space, viewer_state


def load_schemas():
    """
    Load all schemas from the given directory into a dictionary.
    """
    # Get the directory of this file (__init__.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the `schemas` directory
    schema_dir = os.path.join(current_dir, "schemas")

    schemas = {}
    for filename in os.listdir(schema_dir):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            with open(os.path.join(schema_dir, filename), "r") as f:
                schema = yaml.safe_load(f)
                schema_id = schema.get("$id")
                if schema_id:
                    schemas[schema_id] = schema
    return schemas


def validate_json(json_data, schema_id, schemas):
    """
    Validate a JSON document against a schema.

    Parameters:
    - json_data: The JSON document to validate.
    - schema_id: The `$id` of the schema to validate against.
    - schemas: A dictionary of all loaded schemas.

    Raises:
    - ValidationError: If the JSON document does not conform to the schema.
    """
    if schema_id not in schemas:
        raise ValueError(f"Schema with id '{schema_id}' not found.")

    schema = schemas[schema_id]
    resolver = RefResolver(base_uri="", referrer=schema, store=schemas)
    validate(instance=json_data, schema=schema, resolver=resolver)


schemas = load_schemas()


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Args:
        json (dict): A dictionary to be encoded.
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return json.JSONEncoder.default(self, o)


class Annotation(NamedTuple):
    id: int
    encoded: bytes
    relationships: Sequence[Sequence[int]]


_PROPERTY_DTYPES: dict[
    str, tuple[Union[tuple[str], tuple[str, tuple[int, ...]]], int]
] = {
    "uint8": (("|u1",), 1),
    "uint16": (("<u2",), 2),
    "uint32": (("<u4",), 3),
    "int8": (("|i1",), 1),
    "int16": (("<i2",), 2),
    "int32": (("<i4",), 4),
    "float32": (("<f4",), 4),
    "rgb": (("|u1", (3,)), 1),
    "rgba": (("|u1", (4,)), 1),
}

AnnotationType = Literal["point", "line", "axis_aligned_bounding_box", "ellipsoid"]
MINISHARD_TARGET_COUNT = 1000
SHARD_TARGET_SIZE = 50000000


class AnnotationReader:
    def __init__(self, cloudpath: str):
        """Class for reading precomputed annotations from a cloud path.

        Args:
            cloudpath (str): The cloud path to the precomputed annotations. This should be a
                valid tensorstore path, e.g. "gs://bucket/path/to/annotations/".
                use file:// for accessing local files.
        """
        self.cloudpath = cloudpath

        ts_spec = {
            "driver": "json",
            "kvstore": os.path.join(cloudpath, "info"),
        }
        info_future = ts.open(ts_spec).result()
        info = info_future.read().result().item()
        validate_json(info, "PrecomputedAnnotation", schemas)
        self._info = info
        self.info["annotation_type"] = self.info["annotation_type"].lower()
        self.annotation_type = self.info["annotation_type"]

        self._coordinate_space = coordinate_space.CoordinateSpace(
            json=info["dimensions"]
        )

        if "properties" in self.info.keys():
            self._properties = [
                viewer_state.AnnotationPropertySpec(json_data=p)
                for p in self.info["properties"]
            ]
            self._enum_dict = {}
            for p in self._properties:
                if p.enum_labels:
                    self._enum_dict[p.id] = {
                        k: v for k, v in zip(p.enum_values, p.enum_labels)
                    }
        else:
            self._properties = []

        dtype = _get_dtype_for_geometry(
            self.info["annotation_type"], self.coordinate_space.rank
        ) + _get_dtype_for_properties(self.properties)
        dtypeall = np.dtype(dtype)
        self.itemsize = dtypeall.itemsize
        self.dtype = np.dtype(dtype)

        if "by_id" in self.info.keys():
            by_id_info = self.info["by_id"]
            if "sharding" in by_id_info.keys():
                ts_spec = {
                    "base": os.path.join(self.cloudpath, by_id_info["key"]),
                    "driver": "neuroglancer_uint64_sharded",
                    "metadata": by_id_info["sharding"],
                }
                self.by_id_type = "sharded"
            else:
                ts_spec = os.path.join(self.cloudpath, by_id_info["key"]) + "/"
                self.by_id_type = "unsharded"

            self.ts_by_id = ts.KvStore.open(ts_spec).result()

        if "relationships" in self.info.keys():
            self.relationship_ts_dict = {}
            for relationship in self.info["relationships"]:
                name = relationship["id"]
                if "sharding" in relationship.keys():
                    ts_spec = {
                        "base": os.path.join(self.cloudpath, relationship["key"]),
                        "driver": "neuroglancer_uint64_sharded",
                        "metadata": relationship["sharding"],
                    }
                    tstype = "sharded"
                else:
                    ts_spec = os.path.join(self.cloudpath, relationship["key"]) + "/"
                    tstype = "unsharded"

                self.relationship_ts_dict[name] = (
                    ts.KvStore.open(ts_spec).result(),
                    tstype,
                )

        if "spatial" in self.info.keys():
            self.spatial_ts_dict = {}
            for spatial in self.info["spatial"]:
                if "sharding" in spatial.keys():
                    ts_spec = {
                        "base": os.path.join(self.cloudpath, spatial["key"]),
                        "driver": "neuroglancer_uint64_sharded",
                        "metadata": spatial["sharding"],
                    }
                    ts_type = "sharded"
                else:
                    ts_spec = os.path.join(self.cloudpath, spatial["key"]) + "/"
                    ts_type = "unsharded"
                self.spatial_ts_dict[spatial["key"]] = (
                    ts.KvStore.open(ts_spec).result(),
                    ts_type,
                )

    def get_relationships(self):
        """Get the relationships of the annotations.
        Returns:
            list: A list of relationships.
        """
        if "relationships" not in self.info.keys():
            raise ValueError("No relationships found in the info file.")
        return [r["key"] for r in self.info["relationships"]]

    def get_property_names(self):
        """Get the properties of the annotations.

        Returns:
            list: A list of properties.
        """
        if "properties" not in self.info.keys():
            raise ValueError("No properties found in the info file.")
        return [p["id"] for p in self.info["properties"]]

    @property
    def properties(self):
        """Get the properties of the annotations.

        Returns:
            list[neuroglancer.viewer_state.AnnotationPropertySpec]: A list of AnnotationPropertySpec properties.
        """
        return self._properties

    @property
    def coordinate_space(self):
        """Get the coordinate space of the annotations.

        Returns:
            neuroglancer.coordinate_space.CoordinateSpace: The coordinate space.
        """
        return self._coordinate_space

    def get_all_annotation_ids(self):
        """Get all annotation IDs from the kv store.

        Returns:
            np.array: An array of all annotation IDs.
        """
        if self.ts_by_id is None:
            raise ValueError("No by_id information found in the info file.")
        all_ids = self.ts_by_id.list().result()
        if self.by_id_type == "sharded":
            # convert id to binary representation of a uint64
            id_column = np.frombuffer(b"".join(all_ids), dtype=">u8")
        elif self.by_id_type == "unsharded":
            id_column = [int(id_.decode("utf-8")) for id_ in all_ids]
        else:
            raise ValueError(f"Unknown by_id type: {self.by_id_type}")
        return id_column

    def get_all_annotations(self, max_annotations=1_000_000):
        """get all annotations from the kv store. Warning, if the number of annotations is very large
        downloading all the keys will cause large amounts of memory to be used.

        After downloading the keys, this will error if the number of annotations is more than max_annotations.


        Args:
            max_annotations (_type_, optional): Maximum number of annotations to download. Defaults to 1_000_000.


        """

        all_ids = self.ts_by_id.list().result()
        if len(all_ids) > max_annotations:
            raise ValueError(
                f"Number of annotations ({len(all_ids)}) exceeds the maximum allowed ({max_annotations})."
            )
        if self.by_id_type == "sharded":
            id_column = np.frombuffer(b"".join(all_ids), dtype=">u8")
        elif self.by_id_type == "unsharded":
            id_column = all_ids
        futures = [self.ts_by_id.read(id_) for id_ in all_ids]
        # await asyncio.wait(futures, return_when=asyncio.ALL_COMPLETED)
        all_ann_bytes = [b.result() for b in futures]
        anns = [self._decode_annotation(ann_bytes.value) for ann_bytes in all_ann_bytes]
        df = pd.DataFrame(anns)
        df["ID"] = [int(id_) for id_ in id_column]
        df.set_index("ID", inplace=True)

        return self.__post_process_dataframe(df)

    async def iter_all_ann(
        self,
        chunk_size: int = 1_000_000,
        min_key: int = 0,
        max_key: int = 1_000_000_000,
    ):
        """
        Asynchronous generator that yields decoded annotations from `kv_store`
        in chunks, downloading and decoding them in parallel.

        Args:
            chunk_size (int): The number of keys to retrieve in each chunk.
                Default is 1,000,000.
            min_key (int): The starting key. Default is 0.
            max_key (int): The ending key. Default is 1,000,000,000.

        Yields:
            dict: Decoded annotation.
        """
        i = min_key
        while i < max_key:
            start_int = i
            end_int = min(i + chunk_size, max_key)

            # Convert the numeric start & end to the big-endian byte strings
            start_bytes = np.ascontiguousarray(start_int, dtype=">u8").tobytes()
            end_bytes = np.ascontiguousarray(end_int, dtype=">u8").tobytes()

            # Construct the KeyRange for the chunk
            key_range = ts.KvStore.KeyRange(
                inclusive_min=start_bytes, exclusive_max=end_bytes
            )

            # List the keys for this chunk
            keys = await self.ts_by_id.list(key_range)

            if keys:
                # Create asynchronous tasks for downloading and decoding
                tasks = [self._download_and_decode_annotation(key) for key in keys]

                # Process tasks as they complete
                for task in asyncio.as_completed(tasks):
                    annotation = await task
                    yield annotation

            i = end_int

    async def _download_and_decode_annotation(self, key):
        """
        Asynchronously downloads and decodes a single annotation.

        Args:
            key: The key of the annotation to download.

        Returns:
            dict: Decoded annotation.
        """
        # Download the annotation
        result = await self.ts_by_id.read(key)

        # Decode the annotation
        return self._decode_annotation(result.value)

    # Usage example:
    # for key in iter_all_keys(pre_ann.ts_by_id):
    #     do_something_with(key)

    @property
    def info(self):
        return self._info

    def __post_process_dataframe(self, df):
        """Post-process the DataFrame to handle enum properties."""

        for p in self.properties:
            if p.enum_labels:
                df[p.id] = df[p.id].replace(
                    {id: label for id, label in zip(p.enum_values, p.enum_labels)}
                )
        if "geometry" in df.columns:
            if self.annotation_type == "line":
                df["point_a"] = df["geometry"].apply(
                    lambda x: x[: self.coordinate_space.rank]
                )
                df["point_b"] = df["geometry"].apply(
                    lambda x: x[self.coordinate_space.rank :]
                )
            if self.annotation_type == "axis_aligned_bounding_box":
                df["point_a"] = df["geometry"].apply(
                    lambda x: x[: self.coordinate_space.rank]
                )
                df["point_b"] = df["geometry"].apply(
                    lambda x: x[self.coordinate_space.rank :]
                )
            if self.annotation_type == "point":
                df["point"] = df["geometry"]
            if self.annotation_type == "ellipsoid":
                df["center"] = df["geometry"].apply(
                    lambda x: x[: self.coordinate_space.rank]
                )
                df["radii"] = df["geometry"].apply(
                    lambda x: x[self.coordinate_space.rank :]
                )
            df.drop(columns=["geometry"], inplace=True)
        return df

    def decode_multiple_annotations(self, annbytes):
        """Decode multiple annotations from bytes.

        Args:
            annbytes (bytes): The bytes to decode.

        Returns:
            pd.DataFrame: A DataFrame of decoded annotations.
        """
        n_annotations = struct.unpack("<Q", annbytes[:8])[0]
        offset = 8
        ending = offset + n_annotations * self.itemsize
        annarray = np.frombuffer(
            annbytes[offset:ending],
            dtype=self.dtype,
        )
        offset += self.itemsize * n_annotations

        ids = np.frombuffer(annbytes[offset : offset + 8 * n_annotations], dtype="<u8")

        records = []
        for row in annarray:
            record = {
                name: row[name].tolist()
                if isinstance(row[name], np.ndarray)
                else row[name]
                for name in annarray.dtype.names
            }
            records.append(record)

        # Create DataFrame
        df = pd.DataFrame(records, index=pd.Series(ids, name="ID"))

        return self.__post_process_dataframe(df)

    def get_by_relationship(
        self, relationship: str, related_id: int, get_relationships: bool = False
    ):
        """Get annotations by relationship.
        Will not include any relationships this ID has.
        i.e. if the relationship is "parent" and the ID is 1234,
        this will return all annotations that have parent=1234 as a relationship property.

        Args:
            relationship (str): The name of the relationship to filter by based on the info file.
            see self.info['relationships'] for the list of relationships.
            related_id (int): the ID of the relationship.
            get_relationships (bool): If True, will include the relationships in the returned DataFrame.
            This will be slower as it requires downloading each relationship. (Default False)

        Raises:
            ValueError: If relationships are not found in the info file.
            ValueError: If the specific relationship is not found in the info file.

        Returns:
            pd.DataFrame: A DataFrame of annotations that have the specified relationship (does not include relationships)
            index is the annotation ID.
        """
        if "relationships" not in self.info.keys():
            raise ValueError("No relationships found in the info file.")

        # Check if all provided relationships are valid

        if relationship not in [rel["id"] for rel in self.info["relationships"]]:
            raise ValueError(
                f"Invalid relationship '{relationship}' provided. Must be one of {self.relationship_ts_dict.keys()}"
            )

        # Get all annotations
        rel_ts, tstype = self.relationship_ts_dict[relationship]
        if tstype == "sharded":
            key = np.ascontiguousarray(related_id, dtype=">u8").tobytes()
        elif tstype == "unsharded":
            key = str(related_id)
        try:
            annbytes = rel_ts[key]
        except KeyError:
            logging.warning(
                f"No annotations found for relationship '{relationship}' with ID {related_id}. Returning empty DataFrame."
            )
            return self.__empty_df()
        if get_relationships:
            ids = self.__get_multiple_annotation_ids(annbytes)
            return self.get_by_ids(ids)
        return self.decode_multiple_annotations(annbytes)

    def __empty_df(self):
        props = [p.id for p in self.properties]
        relations = [r["id"] for r in self.info["relationships"]]
        df = pd.DataFrame(columns=["ID"] + props + relations)
        return df.set_index("ID")

    def __get_multiple_annotation_ids(self, annbytes):
        """Get the IDs encoded in a multiple annotation bytes encoding.

        Args:
            annbytes (bytes): The bytes to decode.

        Returns:
            pd.DataFrame: A DataFrame of decoded annotations.
        """
        n_annotations = struct.unpack("<Q", annbytes[:8])[0]
        offset = 8 + self.itemsize * n_annotations
        ids = np.frombuffer(annbytes[offset : offset + 8 * n_annotations], dtype="<u8")
        return ids

    def get_by_id(self, id):
        """Get an annotation by its ID. Will include any relationships this ID has.

        Args:
            id (int): The ID of the annotation to retrieve.

        Raises:
            ValueError: If a by_id index is not found in the info file.
            ValueError: If the by_id type is unknown.

        Returns:
            dict: The annotation with the specified ID as a dictionary.
        """
        if "by_id" not in self.info.keys():
            raise ValueError("No by_id information found in the info file.")
        if self.ts_by_id is None:
            raise ValueError("No by_id information found in the info file.")
        if self.by_id_type == "sharded":
            # convert id to binary representation of a uint64
            key = np.ascontiguousarray(id, dtype=">u8").tobytes()
        elif self.by_id_type == "unsharded":
            key = str(id)
        else:
            raise ValueError(f"Unknown by_id type: {self.by_id_type}")
        value = self.ts_by_id[key]
        ann = self._decode_annotation(value)
        for p in self.properties:
            if p.enum_labels:
                ann[p.id] = self._enum_dict[p.id][ann[p.id]]
        return ann

    def get_by_ids(self, ids: Sequence[int]):
        """Get multiple annotations by their IDs.

        Args:
            ids (Sequence[int]): A sequence of IDs of the annotations to retrieve.

        Returns:
            pd.DataFrame: A DataFrame of the annotations with the specified IDs.
        """
        if "by_id" not in self.info.keys():
            raise ValueError("No by_id information found in the info file.")
        if self.ts_by_id is None:
            raise ValueError("No by_id information found in the info file.")

        futures = []
        for id in ids:
            if self.by_id_type == "sharded":
                key = np.ascontiguousarray(id, dtype=">u8").tobytes()
            elif self.by_id_type == "unsharded":
                key = str(id)
            else:
                raise ValueError(f"Unknown by_id type: {self.by_id_type}")
            futures.append(self.ts_by_id.read(key))

        all_ann_bytes = [b.result() for b in futures]
        anns = [self._decode_annotation(ann_bytes.value) for ann_bytes in all_ann_bytes]
        df = pd.DataFrame(anns)
        df["ID"] = ids
        df.set_index("ID", inplace=True)

        return self.__post_process_dataframe(df)

    def _process_geometry(self, ann_dict):
        """Process the geometry of the annotation. Will convert the geometry
        to the appropriate fields based on the annotation type.
        This is used to convert the geometry field in the annotation
        to the appropriate fields for the annotation type.
        For example, for a point annotation, the geometry field will be
        converted to a point column, but for a line it will be converted
        to point_a and point_b.

        Args:
            ann_dict (dict): The annotation dictionary containing the geometry.

        Returns:
            dict: The processed annotation dictionary with geometry fields.
        """
        geom = ann_dict.pop("geometry")
        if self.annotation_type == "line":
            ann_dict["point_a"] = geom[: self.coordinate_space.rank]
            ann_dict["point_b"] = geom[self.coordinate_space.rank :]
        if self.annotation_type == "axis_aligned_bounding_box":
            ann_dict["point_a"] = geom[: self.coordinate_space.rank]
            ann_dict["point_b"] = geom[self.coordinate_space.rank :]
        if self.annotation_type == "point":
            ann_dict["point"] = geom
        if self.annotation_type == "ellipsoid":
            ann_dict["center"] = geom[: self.coordinate_space.rank]
            ann_dict["radii"] = geom[self.coordinate_space.rank :]
        return ann_dict

    def _decode_annotation(self, annbytes):
        """Decode a single annotation from bytes.

        Args:
            annbytes (bytes): The bytes to decode.

        Returns:
            dict: The decoded annotation as a dictionary.
        """

        # Decode the annotation
        dt = self.dtype
        ann_array = np.frombuffer(annbytes[: dt.itemsize], dtype=dt)[0]
        ann_dict = {field: ann_array[field] for field in dt.fields}

        offset = self.itemsize
        for relationship in self.info["relationships"]:
            n_rel = struct.unpack("<I", annbytes[offset : offset + 4])[0]
            relations = np.frombuffer(
                annbytes[offset + 4 : offset + 4 + n_rel * 8], dtype="<Q"
            )
            ann_dict[relationship["id"]] = relations
            offset += 4 + n_rel * 8
        ann_dict = self._process_geometry(ann_dict)

        return ann_dict

    def __get_overlapping_chunks(
        self,
        spatial_key: str,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ):
        """Get the overlapping chunks in a spatial kv store for a given bounding box.

        Args:
            spatial_key (str): The key of the spatial kv store to query.
            lower_bound (Sequence[float]): The lower bound of the bounding box.
            upper_bound (Sequence[float]): The upper bound of the bounding box.

        Raises:
            ValueError: If the spatial key is not found in the info file.
            ValueError: If the length of lower_bound or upper_bound does not match the rank of the coordinate space.

        Returns:
            list[np.array]: a list of tuples, where each tuple represents the indices of a chunk
        """
        if len(lower_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Lower bound length {len(lower_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )
        if len(upper_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Upper bound length {len(upper_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )
        global_lower = np.array(self.info["lower_bound"])
        global_upper = np.array(self.info["upper_bound"])

        local_lower = np.array(lower_bound - global_lower)
        local_upper = np.array(upper_bound - global_lower)

        try:
            spatial_md = next(
                md for md in self.info["spatial"] if md["key"] == spatial_key
            )
        except StopIteration:
            raise ValueError(f"Spatial key '{spatial_key}' not found in the info file.")

        lower_index = np.floor(local_lower / spatial_md["chunk_size"]).astype(int)
        upper_index = np.floor(local_upper / spatial_md["chunk_size"]).astype(int)

        grid_shape = spatial_md.get("grid_shape", spatial_md.get("chunk_shape", None))

        # Ensure lower_index is not greater than upper_index
        lower_index = np.clip(lower_index, 0, np.array(grid_shape) - 1)
        upper_index = np.clip(upper_index, 0, np.array(grid_shape) - 1)

        # Generate all chunk indices in the bounding box
        overlapping_chunks = []

        chunks = product(
            *(
                range(lower_index[i], upper_index[i] + 1)
                for i in range(self.coordinate_space.rank)
            )
        )
        return chunks

    def read_annotations_in_chunk(
        self,
        spatial_key: str,
        chunk_index: Sequence[int],
        get_relationships: bool = False,
        suppress_warning: bool = False,
    ):
        """Read annotations in a specific chunk from the spatial kv store.
           The spatial index metadata can be found at self.info["spatial"]
           which contains a list of downsampling spatial indices.
           Each index has a "key" specified by self.info['spatial']

           Each index contains a grid of chunks that divides the space from
           self.info['lower_bound'] to self.info['upper_bound'] into smaller regions.
           The first index should have the fewest chunks,
           and the last index should have the largest number of chunks.

           Each chunk should only contain self.info["spatial"][i]['max']
           annotations.

           Each chunk is self.info["spatial"][i]['chunk_size'] large.

           So to find the chunk_index of a specific point in the spatial kv store,
           you have to calculate the grid index based on the chunk_size and the lower_bound.
           For example, if the lower bound is [1, 10, 50] and the chunk size is [100, 100, 100],
           and you want to find the chunk index for the point [150, 250, 50],
           you would calculate the chunk index as follows:
              chunk_index = [
                    int((150 - 1) // 100),  # x index
                    int((250 - 10) // 100),  # y index
                    int((50 - 50) // 100)    # z index
                ] = [1, 2, 0]

        Args:
            spatial_key (str): The key of the spatial kv store.
            chunk_index (Sequence[int]): The index of the chunk to read within the grid of the spatial kv store.
            get_relationships (bool): If True, will include the relationships in the returned DataFrame.
            suppress_warning (bool): If True, will suppress warnings about missing chunks.
            Defaults to False.
        Raises:
            ValueError: If the spatial key is not found in the info file.
            ValueError: If the length of chunk_index does not match the rank of the coordinate space.
            ValueError: If the spatial type is unknown.

        Returns:
            pd.DataFrame: DataFrame of annotations in the specified chunk.
        """
        spatial_ts, ts_type = self.spatial_ts_dict.get(spatial_key)
        if spatial_ts is None:
            raise ValueError(f"Spatial key '{spatial_key}' not found in the info file.")

        if len(chunk_index) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected chunk_index to have length {self.coordinate_space.rank}, but received: {len(chunk_index)}"
            )
        spatial_md = next(md for md in self.info["spatial"] if md["key"] == spatial_key)
        if ts_type == "sharded":
            grid_shape = spatial_md.get(
                "grid_shape", spatial_md.get("chunk_shape", None)
            )
            mortoncode = compressed_morton_code(
                chunk_index, np.array(grid_shape, dtype=np.int32)
            )
            chunk_key = np.ascontiguousarray(mortoncode, dtype=">u8").tobytes()
        elif ts_type == "unsharded":
            chunk_key = "_".join([str(c) for c in chunk_index])
        else:
            raise ValueError(f"Unknown spatial type: {ts_type}")
        # Read the chunk from the spatial kv store
        try:
            annbytes = spatial_ts[chunk_key]
        except KeyError:
            # Handle the case where the chunk is not found
            if not suppress_warning:
                logging.warning(
                    f"Chunk {chunk_index} not found in spatial store {spatial_key}. Returning empty DataFrame"
                )
            return self.__empty_df()
        if annbytes == "":
            return self.__empty_df()
        if get_relationships:
            ids = self.__get_multiple_annotation_ids(annbytes)
            return self.get_by_ids(ids)
        # Decode the annotations in the chunk
        return self.decode_multiple_annotations(annbytes)

    def query_spatial_index_in_bounds(
        self,
        spatial_key: str,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ):
        """Query the spatial index to get overlapping chunks in a bounding box.

        Args:
            spatial_key (str): The key of the spatial kv store to query.
            lower_bound (Sequence[float]): The lower bound of the bounding box.
            upper_bound (Sequence[float]): The upper bound of the bounding box.

        Returns:
            list[tuple]: A list of tuples, where each tuple represents the indices of a chunk
        """
        overlapping_chunks = self.__get_overlapping_chunks(
            spatial_key, lower_bound, upper_bound
        )
        dfs = []
        for chunk_index in overlapping_chunks:
            df = self.read_annotations_in_chunk(
                spatial_key, chunk_index, suppress_warning=True
            )
            if not df.empty:
                dfs.append(df)
        if len(dfs) == 0:
            return self.__empty_df()
        return pd.concat(dfs)

    def __post_hoc_spatial_filter(
        self,
        df: pd.DataFrame,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ):
        """filter the annotations to only include the ones in the bounding box.

        Args:
            df (pd.DataFrame): The DataFrame of annotations to filter.
            lower_bound (Sequence[float]): The lower bound of the bounding box.
            upper_bound (Sequence[float]): The upper bound of the bounding box.

        Raises:
            ValueError: If the length of lower_bound or upper_bound does not match the rank of the coordinate space.
            ValueError: If the bounding box is invalid.

        Returns:
            pd.Dataframe: A DataFrame of annotations within the bounding box.
        """

        # Check if the DataFrame is empty
        if df.empty:
            return df

        # Check if the lower and upper bounds are valid
        if len(lower_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Lower bound length {len(lower_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )
        if len(upper_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Upper bound length {len(upper_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )
        # Create a bounding box for the lower and upper bounds
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        bounding_box = np.array([lower_bound, upper_bound])
        # Check if the bounding box is valid
        if np.any(bounding_box[0] > bounding_box[1]):
            raise ValueError(
                "Invalid bounding box: lower bound is greater than upper bound."
            )
        # Filter the DataFrame to only include annotations within the bounding box

        # based on what kind of annotation it is
        # call a different filtering function to figure out
        # if each row of the dataframe is inside the bounding box

        if self.annotation_type == "point":
            points = np.vstack(
                df["point"]
            )  # Convert the "point" column to a 2D NumPy array
            mask = np.all(points >= lower_bound, axis=1) & np.all(
                points <= upper_bound, axis=1
            )
            df = df[mask]
        if self.annotation_type == "line":
            # Combine point_a and point_b into a single 2D array for vectorized comparison
            points_a = np.vstack(df["point_a"])
            points_b = np.vstack(df["point_b"])
            mask = (
                np.all(points_a >= lower_bound, axis=1)
                & np.all(points_a <= upper_bound, axis=1)
            ) | (
                np.all(points_b >= lower_bound, axis=1)
                & np.all(points_b <= upper_bound, axis=1)
            )
            df = df[mask]

        if self.annotation_type == "axis_aligned_bounding_box":
            # Combine point_a and point_b into a single 2D array for vectorized comparison
            points_a = np.vstack(df["point_a"])
            points_b = np.vstack(df["point_b"])
            mask = (
                (
                    np.all(points_a >= lower_bound, axis=1)
                    & np.all(points_a <= upper_bound, axis=1)
                )
                | (
                    np.all(points_b >= lower_bound, axis=1)
                    & np.all(points_b <= upper_bound, axis=1)
                )
                | (
                    np.all(points_a <= lower_bound, axis=1)
                    & np.all(points_b >= upper_bound, axis=1)
                )
                | (
                    np.all(points_b <= lower_bound, axis=1)
                    & np.all(points_a >= upper_bound, axis=1)
                )
            )
            df = df[mask]

        if self.annotation_type == "ellipsoid":
            # Combine center into a single 2D array for vectorized comparison
            centers = np.vstack(df["center"])
            mask = np.all(centers >= lower_bound, axis=1) & np.all(
                centers <= upper_bound, axis=1
            )
            df = df[mask]
        return df

    def get_annotations_in_bounds(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        max_annotations=1_000_000,
    ):
        """Get annotations within a bounding box.

        Args:
            lower_bound (Sequence[float]): The lower bound of the bounding box.
            upper_bound (Sequence[float]): The upper bound of the bounding box.
            max_annotations (int, optional): Maximum number of annotations to return.
                Defaults to 1_000_000.
        Returns:
            pd.DataFrame: DataFrame of annotations within the bounding box.
        """
        if len(lower_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Lower bound length {len(lower_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )
        if len(upper_bound) != self.coordinate_space.rank:
            raise ValueError(
                f"Upper bound length {len(upper_bound)} does not match coordinate space rank {self.coordinate_space.rank}."
            )

        if self.spatial_ts_dict is None:
            raise ValueError("No spatial information found in the info file.")

        total_annotations = 0
        dfs = []
        for spatial_key in self.spatial_ts_dict.keys():
            df = self.query_spatial_index_in_bounds(
                spatial_key, lower_bound, upper_bound
            )
            total_annotations += len(df)
            dfs.append(df)
            if total_annotations > max_annotations:
                logging.warning(
                    f"Exceeded maximum annotations limit: {max_annotations}. Returning partial results."
                )
                break
        df = pd.concat(dfs) if dfs else self.__empty_df()
        if total_annotations > max_annotations:
            df = df.head(max_annotations)

        return self.__post_hoc_spatial_filter(df, lower_bound, upper_bound)


class ShardSpec(NamedTuple):
    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self):
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def choose_output_spec(
    total_count,
    total_bytes,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress=True,
):
    if total_count == 1:
        return None
    if ts is None:
        return None

    # test if hashtype is valid
    if hashtype not in ["murmurhash3_x86_128", "identity_hash"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity_hash'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)
    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )


def morton_code_to_gridpt(code, grid_size):
    gridpt = np.zeros(
        [
            3,
        ],
        dtype=int,
    )

    num_bits = [math.ceil(math.log2(size)) for size in grid_size]
    j = np.uint64(0)
    one = np.uint64(1)

    if sum(num_bits) > 64:
        raise ValueError(
            f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits."
        )

    max_coords = np.max(gridpt, axis=0)
    if np.any(max_coords >= grid_size):
        raise ValueError(
            f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}"
        )

    code = np.uint64(code)

    for i in range(max(num_bits)):
        for dim in range(3):
            i = np.uint64(i)
            if 2**i < grid_size[dim]:
                bit = np.uint64((code >> j) & one)
                gridpt[dim] += bit << i
                j += one

    return gridpt


def compressed_morton_code(gridpt, grid_size):
    """Converts a grid point to a compressed morton code.
    from cloud-volume"""
    if hasattr(gridpt, "__len__") and len(gridpt) == 0:  # generators don't have len
        return np.zeros((0,), dtype=np.uint32)

    gridpt = np.asarray(gridpt, dtype=np.uint32)
    single_input = False
    if gridpt.ndim == 1:
        gridpt = np.atleast_2d(gridpt)
        single_input = True

    code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
    num_bits = [math.ceil(math.log2(size)) for size in grid_size]
    j = np.uint64(0)
    one = np.uint64(1)

    if sum(num_bits) > 64:
        raise ValueError(
            f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits."
        )

    max_coords = np.max(gridpt, axis=0)
    if np.any(max_coords >= grid_size):
        raise ValueError(
            f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}"
        )

    for i in range(max(num_bits)):
        for dim in range(3):
            if 2**i < grid_size[dim]:
                bit = ((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j
                code |= bit
                j += one

    if single_input:
        return code[0]
    return code


def _get_dtype_for_geometry(annotation_type: AnnotationType, rank: int):
    geometry_size = rank if annotation_type == "point" else 2 * rank
    return [("geometry", "<f4", geometry_size)]


def _get_dtype_for_properties(
    properties: Sequence[viewer_state.AnnotationPropertySpec],
):
    dtype = []
    offset = 0
    for i, p in enumerate(properties):
        dtype_entry, alignment = _PROPERTY_DTYPES[p.type]
        # if offset % alignment:
        #     padded_offset = (offset + alignment - 1) // alignment * alignment
        #     padding = padded_offset - offset
        #     dtype.append((f"padding{offset}", "|u1", (padding,)))
        #     offset += padding
        dtype.append((f"{p.id}", *dtype_entry))  # type: ignore[arg-type]
        size = np.dtype(dtype[-1:]).itemsize
        offset += size
    alignment = 4
    if offset % alignment:
        padded_offset = (offset + alignment - 1) // alignment * alignment
        padding = padded_offset - offset
        dtype.append((f"padding{offset}", "|u1", (padding,)))
        offset += padding
    return dtype


class AnnotationWriter:
    annotations: list[Annotation]
    related_annotations: list[dict[int, list[Annotation]]]

    def __init__(
        self,
        annotation_type: AnnotationType,
        names: Sequence[str] = ["x", "y", "z"],
        scales: Sequence[float] = (1.0, 1.0, 1.0),
        units: [str, Sequence[str]] = "nm",
        relationships: Sequence[str] = (),
        properties: Sequence[viewer_state.AnnotationPropertySpec] = (),
        experimental_chunk_size: Union[float, Sequence[float]] = 256,
    ):
        """Initializes an `AnnotationWriter`.

        Args:

            annotation_type: The type of annotation.  Must be one of "point",
                "line", "axis_aligned_bounding_box", or "ellipsoid".
            names : The names of the dimensions of the coordinate space.
                    Default is ["x", "y", "z"], should match whatever
                    axes names you are trying to visualize in neuroglancer.
            scales: The scales of the dimensions of the coordinate space.
                Default is (1.0, 1.0, 1.0). Needs to match length of names
            units: The units of the dimensions of the coordinate space.
                Can be a list, or if all dimensions are a single unit, a single string.
                Default is "nm". Must be a combo of a valid SI unit ["m", "s", "rad/s", "Hz"]
                with a valid prefix ["Y", "Z", "E", "P", "T", "G", "M", "k", "h", "", "c", "m", "u", "µ", "n", "p", "f", "a", "z", "y"]
                If a list it needs to match length of names.
            lower_bound: The lower bound of the bounding box of the annotations.
            relationships: The names of relationships between annotations.  Each
                relationship is a string that is used as a key in the `relationships`
                field of each annotation.  For example, if `relationships` is
                `["parent", "child"]`, then each annotation may have a `parent` and
                `child` relationship, and the `relationships` field of each annotation
                is a dictionary with keys `"parent"` and `"child"`.
            properties: The properties of each annotation.  Each property is a
                `AnnotationPropertySpec` object.
            experimental_chunk_size: The size of each chunk in the spatial index.
                If an integer then all dimensions will be the same chunk size.
                If a sequence, then must have the same length as `coordinate_space.rank`.
                NOTE: it is anticipated that in the future downsampling will be added which
                will start at a single top level chunk and move down, at which time this parameter
                will be removed in favor of parameters that control downsampling.
        """
        if isinstance(units, str):
            units = [units] * len(names)

        if len(names) != len(scales):
            raise ValueError(
                f"Expected names and scales to have the same length, but received: {len(names)} and {len(scales)}"
            )
        if len(names) != len(units):
            raise ValueError(
                f"Expected names and units to have the same length, but received: {len(names)} and {len(units)}"
            )

        self.coordinate_space = CoordinateSpace(
            names=names,
            scales=scales,
            units=units,
        )
        self.relationships = list(relationships)
        self.annotation_type = annotation_type
        self.properties = list(properties)
        self.annotations_by_chunk: defaultdict[str, list[Annotation]] = defaultdict(
            list
        )
        self.properties.sort(key=lambda p: -_PROPERTY_DTYPES[p.type][1])
        self.annotations = []
        self.rank = self.coordinate_space.rank
        self.dtype = _get_dtype_for_geometry(
            annotation_type, self.coordinate_space.rank
        ) + _get_dtype_for_properties(self.properties)

        # if chunk_size is an integer, then make it a sequence
        if isinstance(experimental_chunk_size, numbers.Real):
            self.chunk_size = np.full(
                shape=(self.rank,), fill_value=experimental_chunk_size, dtype=np.float64
            )
        else:
            chunk_size = cast(Sequence[float], experimental_chunk_size)
            if len(chunk_size) != self.rank:
                raise ValueError(
                    f"Expected experimental_chunk_size to have length {self.rank}, but received: {chunk_size}"
                )
            self.chunk_size = np.array(chunk_size)

        self.lower_bound = np.full(
            shape=(self.rank,), fill_value=float("inf"), dtype=np.float32
        )
        self.upper_bound = np.full(
            shape=(self.rank,), fill_value=float("-inf"), dtype=np.float32
        )
        self.related_annotations = [{} for _ in self.relationships]
        p = rtree.index.Property()
        p.dimension = self.rank
        self.rtree = rtree.index.Index(properties=p)

    def get_chunk_index(self, coords):
        return tuple((coords // self.chunk_size).astype(np.int32))

    def add_point(self, point: Sequence[float], id: Optional[int] = None, **kwargs):
        if self.annotation_type != "point":
            raise ValueError(
                f"Expected annotation type point, but received: {self.annotation_type}"
            )
        if len(point) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected point to have length {self.coordinate_space.rank}, but received: {len(point)}"
            )

        self._add_obj(point, id, np.array(point), np.array(point), **kwargs)

    def add_axis_aligned_bounding_box(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        if self.annotation_type != "axis_aligned_bounding_box":
            raise ValueError(
                f"Expected annotation type axis_aligned_bounding_box, but received: {self.annotation_type}"
            )
        lower_bound = np.minimum(point_a, point_b)
        upper_bound = np.maximum(point_a, point_b)
        self._add_two_point_obj(
            point_a, point_b, lower_bound, upper_bound, id, **kwargs
        )

    def add_ellipsoid(
        self,
        center: Sequence[float],
        radii: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        if self.annotation_type != "ellipsoid":
            raise ValueError(
                f"Expected annotation type ellipsoid, but received: {self.annotation_type}"
            )
        if len(center) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected center to have length {self.coordinate_space.rank}, but received: {len(center)}"
            )

        if len(radii) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected radii to have length {self.coordinate_space.rank}, but received: {len(radii)}"
            )

        lower_bound = np.array(center) - np.array(radii)
        upper_bound = np.array(center) + np.array(radii)
        self._add_two_point_obj(center, radii, lower_bound, upper_bound, id, **kwargs)

    def add_line(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        if self.annotation_type != "line":
            raise ValueError(
                f"Expected annotation type line, but received: {self.annotation_type}"
            )
        lower_bound = np.minimum(point_a, point_b)
        upper_bound = np.maximum(point_a, point_b)

        self._add_two_point_obj(
            point_a, point_b, lower_bound, upper_bound, id, **kwargs
        )

    def _add_two_point_obj(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        id: Optional[int] = None,
        **kwargs,
    ):
        if len(point_a) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected coordinates to have length {self.coordinate_space.rank}, but received: {len(point_a)}"
            )

        if len(point_b) != self.coordinate_space.rank:
            raise ValueError(
                f"Expected coordinates to have length {self.coordinate_space.rank}, but received: {len(point_b)}"
            )

        coords = np.concatenate((point_a, point_b))
        self._add_obj(
            cast(Sequence[float], coords),
            id,
            lower_bound=upper_bound,
            upper_bound=upper_bound,
            **kwargs,
        )

    def _add_obj(
        self,
        coords: Sequence[float],
        id: Optional[int],
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        **kwargs,
    ):
        self.lower_bound = np.minimum(self.lower_bound, lower_bound)
        self.upper_bound = np.maximum(self.upper_bound, upper_bound)

        encoded = np.zeros(shape=(), dtype=self.dtype)
        encoded[()]["geometry"] = coords

        for i, p in enumerate(self.properties):
            if p.id in kwargs:
                # encoded[()][f"property{i}"] = kwargs.pop(p.id)
                encoded[()][p.id] = kwargs.pop(p.id)

        related_ids = []
        for relationship in self.relationships:
            ids = kwargs.pop(relationship, None)
            if ids is None:
                ids = []
            if isinstance(ids, numbers.Integral):
                ids = [ids]
            related_ids.append(ids)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments {kwargs}")

        if id is None:
            id = len(self.annotations)

        annotation = Annotation(
            id=id, encoded=encoded.tobytes(), relationships=related_ids
        )

        self.rtree.insert(id, tuple(lower_bound) + tuple(upper_bound), obj=annotation)

        self.annotations.append(annotation)
        for i, segment_ids in enumerate(related_ids):
            for segment_id in segment_ids:
                rel_index = self.related_annotations[i]
                rel_index_list = rel_index.setdefault(segment_id, [])
                rel_index_list.append(annotation)

    def _serialize_annotations_sharded(self, path, annotations, shard_spec):
        spec = {
            "driver": "neuroglancer_uint64_sharded",
            "metadata": shard_spec.to_json(),
            "base": f"file://{path}",
        }
        dataset = ts.KvStore.open(spec).result()
        txn = ts.Transaction()
        for ann in annotations:
            # convert the ann.id to a binary representation of a uint64
            key = np.ascontiguousarray(ann.id, dtype=">u8").tobytes()
            value = ann.encoded
            for related_ids in ann.relationships:
                value += struct.pack("<I", len(related_ids))
                for related_id in related_ids:
                    value += struct.pack("<Q", related_id)
            dataset.with_transaction(txn)[key] = value
        txn.commit_async().result()

    def _serialize_annotations(self, f, annotations: list[Annotation]):
        f.write(self._encode_multiple_annotations(annotations))

    def _serialize_annotation(self, f, annotation: Annotation):
        f.write(annotation.encoded)
        for related_ids in annotation.relationships:
            f.write(struct.pack("<I", len(related_ids)))
            for related_id in related_ids:
                f.write(struct.pack("<Q", related_id))

    def _encode_multiple_annotations(self, annotations: list[Annotation]):
        """
        This function creates a binary string from a list of annotations.

        Parameters:
            annotations (list): List of annotation objects. Each object should have 'encoded' and 'id' attributes.

        Returns:
            bytes: Binary string of all components together.
        """
        binary_components = []
        binary_components.append(struct.pack("<Q", len(annotations)))
        for annotation in annotations:
            binary_components.append(annotation.encoded)
        for annotation in annotations:
            binary_components.append(struct.pack("<Q", annotation.id))
        return b"".join(binary_components)

    def _serialize_annotations_by_related_id(self, path, related_id_dict, shard_spec):
        spec = {
            "driver": "neuroglancer_uint64_sharded",
            "metadata": shard_spec.to_json(),
            "base": f"file://{path}",
        }
        dataset = ts.KvStore.open(spec).result()
        txn = ts.Transaction()
        for related_id, annotations in related_id_dict.items():
            # convert the related_id to a binary representation of a uint64
            key = np.ascontiguousarray(related_id, dtype=">u8").tobytes()
            value = self._encode_multiple_annotations(annotations)
            dataset.with_transaction(txn)[key] = value
        txn.commit_async().result()

    def _serialize_annotation_chunk_sharded(self, path, shard_spec, max_sizes):
        spec = {
            "driver": "neuroglancer_uint64_sharded",
            "metadata": shard_spec.to_json(),
            "base": f"file://{path}",
        }
        dataset = ts.KvStore.open(spec).result()
        txn = ts.Transaction()

        # Generate all combinations of coordinates
        coordinates = product(*(range(n) for n in max_sizes))

        # Iterate over the grid
        for cell in coordinates:
            # Query the rtree index for annotations in the current chunk
            lower_bound = self.lower_bound + np.array(cell) * self.chunk_size
            upper_bound = lower_bound + self.chunk_size
            coords = np.concatenate((lower_bound, upper_bound))
            chunk_annotations = list(
                self.rtree.intersection(tuple(coords), objects="raw")
            )
            if len(chunk_annotations) > 0:
                key = compressed_morton_code(cell, max_sizes)
                # convert the np.uint64 to a binary representation of a uint64
                # using big endian representation
                key = np.ascontiguousarray(key, dtype=">u8").tobytes()
                value = self._encode_multiple_annotations(chunk_annotations)
                dataset.with_transaction(txn)[key] = value

        txn.commit_async().result()

    def write(self, path: Union[str, pathlib.Path], write_sharded: bool = True):
        metadata = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": self.coordinate_space.to_json(),
            "lower_bound": [float(x) for x in self.lower_bound],
            "upper_bound": [float(x) for x in self.upper_bound],
            "annotation_type": self.annotation_type,
            "properties": [p.to_json() for p in self.properties],
            "relationships": [],
            "by_id": {"key": "by_id"},
        }
        total_ann_bytes = sum(len(a.encoded) for a in self.annotations)
        sharding_spec = choose_output_spec(len(self.annotations), total_ann_bytes)

        # calculate the number of chunks in each dimension
        num_chunks = np.ceil(
            (self.upper_bound - self.lower_bound) / self.chunk_size
        ).astype(int)

        num_chunks = np.maximum(num_chunks, np.full(num_chunks.shape, 1, dtype=int))

        metadata["upper_bound"] = self.lower_bound + (num_chunks * self.chunk_size)
        # make directories
        os.makedirs(path, exist_ok=True)
        for relationship in self.relationships:
            os.makedirs(os.path.join(path, f"rel_{relationship}"), exist_ok=True)
        os.makedirs(os.path.join(path, "by_id"), exist_ok=True)
        os.makedirs(os.path.join(path, "spatial0"), exist_ok=True)

        total_chunks = len(self.annotations_by_chunk)
        spatial_sharding_spec = choose_output_spec(
            total_chunks, total_ann_bytes + 8 * len(self.annotations) + 8 * total_chunks
        )
        # initialize metadata for spatial index
        metadata["spatial"] = [
            {
                "key": "spatial0",
                "grid_shape": num_chunks.tolist(),
                "chunk_size": [int(x) for x in self.chunk_size],
                "limit": len(self.annotations),
            }
        ]
        spatial_sharding_spec = None
        # write annotations by spatial chunk
        if (spatial_sharding_spec is not None) and write_sharded:
            self._serialize_annotation_chunk_sharded(
                os.path.join(path, "spatial0"),
                spatial_sharding_spec,
                num_chunks.tolist(),
            )
            metadata["spatial"][0]["sharding"] = spatial_sharding_spec.to_json()
        else:
            # Generate all combinations of coordinates
            coordinates = product(*(range(n) for n in num_chunks))

            # Iterate over the grid
            for cell in coordinates:
                # Query the rtree index for annotations in the current chunk
                lower_bound = self.lower_bound + np.array(cell) * self.chunk_size
                upper_bound = lower_bound + self.chunk_size
                coords = np.concatenate((lower_bound, upper_bound))
                chunk_annotations = list(
                    self.rtree.intersection(tuple(coords), objects="raw")
                )
                if len(chunk_annotations) > 0:
                    chunk_name = "_".join([str(c) for c in cell])
                    filepath = os.path.join(path, "spatial0", chunk_name)
                    with open(filepath, "wb") as f:
                        self._serialize_annotations(f, chunk_annotations)

        # write annotations by id
        if (sharding_spec is not None) and write_sharded:
            self._serialize_annotations_sharded(
                os.path.join(path, "by_id"), self.annotations, sharding_spec
            )
            metadata["by_id"]["sharding"] = sharding_spec.to_json()
        else:
            for annotation in self.annotations:
                with open(os.path.join(path, "by_id", str(annotation.id)), "wb") as f:
                    self._serialize_annotation(f, annotation)

        # write relationships
        for i, relationship in enumerate(self.relationships):
            rel_index = self.related_annotations[i]
            relationship_sharding_spec = choose_output_spec(
                len(rel_index),
                total_ann_bytes + 8 * len(self.annotations) + 8 * total_chunks,
            )
            rel_md = {"id": relationship, "key": f"rel_{relationship}"}
            if (relationship_sharding_spec is not None) and write_sharded:
                rel_md["sharding"] = relationship_sharding_spec.to_json()
                self._serialize_annotations_by_related_id(
                    os.path.join(path, f"rel_{relationship}"),
                    rel_index,
                    relationship_sharding_spec,
                )
            else:
                for segment_id, annotations in rel_index.items():
                    filepath = os.path.join(
                        path, f"rel_{relationship}", str(segment_id)
                    )
                    with open(filepath, "wb") as f:
                        self._serialize_annotations(f, annotations)

            metadata["relationships"].append(rel_md)

        # write metadata info file
        with open(os.path.join(path, "info"), "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, cls=NumpyEncoder))
