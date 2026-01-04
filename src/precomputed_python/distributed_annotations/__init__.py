# @license
# Copyright 2025 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed framework for generating large-scale precomputed annotation datasets.

This framework enables generation of annotation collections with billions of rows
by processing data in parallel without requiring all data to be held in memory.
It fully implements the multi-level spatial index specification with proper
handling of global constraints.

Key features:
- Distributed processing using Dask for out-of-core operations
- TensorStore integration for writing directly to cloud storage (GCS, S3, etc.)
- Full implementation of multi-level spatial index with proper sampling
- Support for both sharded and unsharded index formats
- Efficient handling of annotations that span multiple spatial cells
"""

from .encoder import AnnotationEncoder
from .spatial_index import SpatialIndexBuilder
from .writer import DistributedAnnotationWriter
from .pipeline import AnnotationPipeline

__all__ = [
    "AnnotationEncoder",
    "SpatialIndexBuilder",
    "DistributedAnnotationWriter",
    "AnnotationPipeline",
]

