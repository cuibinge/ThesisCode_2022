import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

Potsdam_CATEGORIES = [
    {"color": [255, 255, 255], "id": 1, "name": "impervious surfaces", "trainId": 0},
    {"color": [0, 0, 255], "id": 2, "name": "building", "trainId": 1},
    {"color": [0, 255, 255], "id": 3, "name": "low vegetation", "trainId": 2},
    {"color": [0, 255, 0], "id": 4, "name": "tree", "trainId": 3},
    {"color": [255, 255, 0], "id": 5, "name": "car", "trainId": 4},
]


def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in Potsdam_CATEGORIES]


    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in Potsdam_CATEGORIES]
    stuff_color = [k["color"] for k in Potsdam_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_color,
    }

    return ret

def register_Potsdam(root):
    root = os.path.join(root, "Potsdam")
    meta = _get_coco_stuff_meta()
    print(meta["stuff_colors"])
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"Potsdam_{name}"
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
register_Potsdam(_root)
