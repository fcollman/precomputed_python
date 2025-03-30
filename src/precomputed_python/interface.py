import precomputed_python

pp = precomputed_python.AnnotationReader(
    source="gs://neuroglancer-20191211_fafbv14_buhmann2019_li20190805"
)

pp.get_properties()
# ['cleft_score', 'score', 'id', 'autapse']
pp.get_annotation_type()
# 'line'
pp.get_relationships()
# ['pre_segment', 'post_segment']
pp.get_dimensions()
# np.array([1.0, 1.0, 1.0]), ['nm','nm','nm'], ['x','y','z']

pp.get_by_id(id=141187882)
# {'ID': 167924347,
#  'point_a'=[109709, 40322, 5357],
#  'point_b'=[109722, 40308, 5357],
#  'cleft_score': 153.000,
#  'score': 803.222,
#  'autapse': 0,
#  'id': 167924347,
#  'pre_segment': 710435991,
#  'post_segment': 10214831972}

df = pp.get_by_relationship(pre_segment=710435991)
# df.head()
# point_a,point_b,cleft_score,score,autapse,id,pre_segment,post_segment
# 167924347,[109709, 40322, 5357],[109722, 40308, 5357],153.000 803.222, 0,167924347, 710435991,10214831972
# ...
# ...

df = pp.get_by_bounding_box(
    lower_bound=[38904, 36396, 40],
    upper_bound=np.array([38904, 36396, 40]) + [10000, 10000, 100],
)
# df.head()
# point_a,point_b,cleft_score,score,autapse,id,pre_segment,post_segment
# ...
# ...
# ...
# ...
coord_space = neuroglancer.CoordinateSpace(
    names=["x", "y", "z"], units=["nm"] * 3, scales=[1, 1, 1]
)
properties = ["cleft_score", "score", "id", "autapse"]
props = []
props.append(
    neuroglancer.viewer_state.AnnotationPropertySpec(
        id="cleft_score", type="float32", description="the score of the cleft"
    )
)
props.append(
    neuroglancer.viewer_state.AnnotationPropertySpec(
        id="score", type="float32", description="the score of the annotation"
    )
)
props.append(
    neuroglancer.viewer_state.AnnotationPropertySpec(
        id="id", type="uint64", description="the id of the annotation"
    )
)
props.append(
    neuroglancer.viewer_state.AnnotationPropertySpec(
        id="autapse", type="int8", description="the id of the annotation"
    )
)

pp = precomputed_python.AnnotationWriter(
    source="gs://neuroglancer-20191211_fafbv14_buhmann2019_li20190805",
    coordinate_space=coord_space,
    relationships=["pre_segment", "post_segment"],
    properties=props,
    annotation_type="line",
    experimental_chunk_size=[1024, 1024, 1024],
)
