import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

huang_CATEGORIES = [
    {"color": [0, 92, 230], "id": 1, "name": "Seagrass bed", "trainId": 0},
    {"color": [255, 0, 0], "id": 2, "name": "Spartina alterniflora", "trainId": 1},
    {"color": [171, 255, 0], "id": 3, "name": "Reed", "trainId": 2},
    {"color": [23, 227, 36], "id": 4, "name": "Tamarix", "trainId": 3},
    {"color": [240, 205, 111], "id": 5, "name": "Tidal flat", "trainId": 4},
    {"color": [229, 153, 0], "id": 6, "name": "Sparse vegetation", "trainId": 5},
    {"color": [97, 255, 247], "id": 7, "name": "Pond", "trainId": 6},
    {"color": [119, 177, 252], "id": 8, "name": "Yellow River", "trainId": 7},
    {"color": [1, 133, 251], "id": 9, "name": "Sea", "trainId": 8},
    {"color": [255, 255, 255], "id": 10, "name": "Cloud", "trainId": 9}
]


def _get_coco_stuff_meta():
    stuff_ids = [k["id"] for k in huang_CATEGORIES]


    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in huang_CATEGORIES]
    stuff_color = [k["color"] for k in huang_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_color,
    }

    return ret

def register_huang(root):
    root = os.path.join(root, "huang")
    meta = _get_coco_stuff_meta()
    print(meta["stuff_colors"])
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"huang_{name}"
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
register_huang(_root)
