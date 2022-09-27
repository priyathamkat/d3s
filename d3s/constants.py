from pathlib import Path

IMAGENET_PATH = Path("/fs/cml-datasets/ImageNet/ILSVRC2012/")
SALIENT_IMAGENET_PATH = Path("/fs/cml-datasets/Salient-ImageNet/")
SALIENT_IMAGENET_MASKS_PATH = SALIENT_IMAGENET_PATH / "salient_imagenet_dataset"
MTURK_RESULTS_CSV_PATH = (
    SALIENT_IMAGENET_PATH
    / "salient_imagenet-main/mturk_results/discover_spurious_features.csv"
)
COCO_ROOT = Path("/fs/cml-datasets/coco/")
