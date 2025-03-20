import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

LoveDA_CATEGORIES = [
    {"color": [255, 0, 0], "id": 2, "name": "building", "trainId": 0},
    {"color": [255, 255, 0], "id": 3, "name": "road", "trainId": 1},
    {"color": [0, 0, 255], "id": 4, "name": "water", "trainId": 2},
    {"color": [159, 129, 183], "id": 5, "name": "barren", "trainId": 3},
    {"color": [0, 255, 0], "id": 6, "name": "forest", "trainId": 4},
    {"color": [255, 195, 128], "id": 7, "name": "agriculture", "trainId": 5},
]


def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in LoveDA_CATEGORIES]


    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in LoveDA_CATEGORIES]
    stuff_color = [k["color"] for k in LoveDA_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_color,
    }

    return ret

def register_LoveDA(root):
    root = os.path.join(root, "LoveDA")
    meta = _get_coco_stuff_meta()
    print(meta["stuff_colors"])
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"LoveDA_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
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
register_LoveDA(_root)
