import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FBP_CATEGORIES = [
    {"color": [200, 0, 0], "isthing": 1, "id": 1, "name": "industrial area", "trainId": 0},


    {"color": [150, 200, 150], "isthing": 1, "id": 4, "name": "dry cropland", "trainId": 1},
    {"color": [200, 0, 200], "isthing": 1, "id": 5, "name": "garden land", "trainId": 2},
    {"color": [150, 0, 250], "isthing": 1, "id": 6, "name": "arbor forest", "trainId": 3},
    {"color": [150, 150, 250], "isthing": 1, "id": 7, "name": "shrub forest", "trainId": 4},
    {"color": [200, 150, 200], "isthing": 1, "id": 8, "name": "park", "trainId": 5},
    {"color": [250, 200, 0], "isthing": 1, "id": 9, "name": "natural meadow", "trainId": 6},
    {"color": [200, 200, 0], "isthing": 1, "id": 10, "name": "artificial meadow", "trainId": 7},

    {"color": [250, 0, 150], "isthing": 1, "id": 12, "name": "urban residential", "trainId": 8},
    {"color": [0, 150, 200], "isthing": 1, "id": 13, "name": "lake", "trainId": 9},

    {"color": [150, 200, 250], "isthing": 1, "id": 15, "name": "fish pond", "trainId": 10},
    {"color": [250, 250, 250], "isthing": 1, "id": 16, "name": "snow", "trainId": 11},
    {"color": [200, 200, 200], "isthing": 1, "id": 17, "name": "bareland", "trainId": 12},

    {"color": [250, 200, 150], "isthing": 1, "id": 19, "name": "stadium", "trainId": 13},
    {"color": [150, 150, 0], "isthing": 1, "id": 20, "name": "square", "trainId": 14},

    {"color": [250, 150, 0], "isthing": 1, "id": 22, "name": "overpass", "trainId": 15},
    {"color": [250, 200, 250], "isthing": 1, "id": 23, "name": "railway station", "trainId": 16},
    {"color": [200, 150, 0], "isthing": 1, "id": 24, "name": "airport", "trainId": 17}
]

def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in FBP_CATEGORIES]
    assert len(stuff_ids) == 18, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in FBP_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_FBP_18(root):
    root = os.path.join(root, "FBP18")
    meta = _get_coco_stuff_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"FBP_18_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_FBP_18(_root)
