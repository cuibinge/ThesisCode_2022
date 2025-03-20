import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FBP_CATEGORIES = [
  {"color": [200, 0, 0], "isthing": 1, "id": 1, "name": "industrial area"},
  {"color": [0, 200, 0], "isthing": 1, "id": 2, "name": "paddy field"},
  {"color": [150, 250, 0], "isthing": 1, "id": 3, "name": "irrigated field"},
  {"color": [150, 200, 150], "isthing": 1, "id": 4, "name": "dry cropland"},
  {"color": [200, 0, 200], "isthing": 1, "id": 5, "name": "garden land"},
  {"color": [150, 0, 250], "isthing": 1, "id": 6, "name": "arbor forest"},
  {"color": [150, 150, 250], "isthing": 1, "id": 7, "name": "shrub forest"},
  {"color": [200, 150, 200], "isthing": 1, "id": 8, "name": "park"},
  {"color": [250, 200, 0], "isthing": 1, "id": 9, "name": "natural meadow"},
  {"color": [200, 200, 0], "isthing": 1, "id": 10, "name": "artificial meadow"},
  {"color": [0, 0, 200], "isthing": 1, "id": 11, "name": "river"},
  {"color": [250, 0, 150], "isthing": 1, "id": 12, "name": "urban residential"},
  {"color": [0, 150, 200], "isthing": 1, "id": 13, "name": "lake"},
  {"color": [0, 200, 250], "isthing": 1, "id": 14, "name": "pond"},
  {"color": [150, 200, 250], "isthing": 1, "id": 15, "name": "fish pond"},
  {"color": [250, 250, 250], "isthing": 1, "id": 16, "name": "snow"},
  {"color": [200, 200, 200], "isthing": 1, "id": 17, "name": "bareland"},
  {"color": [200, 150, 150], "isthing": 1, "id": 18, "name": "rural residential"},
  {"color": [250, 200, 150], "isthing": 1, "id": 19, "name": "stadium"},
  {"color": [150, 150, 0], "isthing": 1, "id": 20, "name": "square"},
  {"color": [250, 150, 150], "isthing": 1, "id": 21, "name": "road"},
  {"color": [250, 150, 0], "isthing": 1, "id": 22, "name": "overpass"},
  {"color": [250, 200, 250], "isthing": 1, "id": 23, "name": "railway station"},
  {"color": [200, 150, 0], "isthing": 1, "id": 24, "name": "airport"}
]


def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in FBP_CATEGORIES]
    assert len(stuff_ids) == 24, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in FBP_CATEGORIES]
    stuff_color = [k["color"] for k in FBP_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_color,
    }

    return ret

def register_FBP_24(root):
    root = os.path.join(root, "FBP24")
    meta = _get_coco_stuff_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"FBP_24_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=meta["stuff_classes"],
            stuff_colors=meta["stuff_colors"],

        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_FBP_24(_root)
