from pathlib import Path

import timm
import torch
from absl import app, flags
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3s.datasets import D3S, ImageNet
from d3s.pretrained_models import DINO, CLIPZeroShotClassifier, RobustResNet50

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluating the model")
flags.DEFINE_string(
    "output_folder",
    "/cmlscratch/pkattaki/void/d3s/d3s/outputs/d3s_eval_stats",
    "Folder to save the outputs",
)
flags.DEFINE_integer("rank", 0, "Rank of GPU to use")
flags.DEFINE_integer("num_workers", 2, "Number of workers to use")
flags.DEFINE_list("model_names", [], "List of model names to evaluate")


@torch.no_grad()
def test(model, dataloader, device, pbar, is_d3s=False, num_classes=0, num_bgs=0):
    model.eval()
    top1 = top5 = total = 0
    # stats: {top1, top5, total} X classes X backgrounds
    stats = torch.zeros(3, num_classes, num_bgs) if is_d3s else None
    for batch in dataloader:
        images, class_idxs = batch[:2]
        images = images.to(device)

        outputs = model(images)
        _, top5_preds = outputs.topk(5, dim=1)
        top5_preds = top5_preds.cpu()
        top1_preds = top5_preds[:, 0]

        is_correct_top1 = (top1_preds == class_idxs).float()
        is_correct_top5 = (top5_preds == class_idxs.unsqueeze(1)).any(dim=1).float()

        if is_d3s:
            bg_idxs = batch[2]

            stats[0, class_idxs, bg_idxs] += is_correct_top1
            stats[1, class_idxs, bg_idxs] += is_correct_top5
            stats[2, class_idxs, bg_idxs] += 1

        top5 += is_correct_top5.sum().item()
        top1 += is_correct_top1.sum().item()
        batch_size = class_idxs.shape[0]
        total += batch_size
        pbar.update(batch_size)

    return 100 * top1 / total, 100 * top5 / total, stats


def evaluate_model(model_name, output_folder, pbar):
    device = torch.device(f"cuda:{FLAGS.rank}")

    if model_name.startswith("dino"):
        model = DINO(model_name)
        transform = model.transform
    elif model_name.startswith("clip"):
        model = CLIPZeroShotClassifier(model_name, device)
        transform = model.transform
    elif model_name.startswith("robust"):
        model = RobustResNet50(model_name)
        transform = model.transform
    else:
        model = timm.create_model(model_name, pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    model.to(device)

    dataset = ImageNet(split="val", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    imagenet_top1, imagenet_top5, _ = test(
        model, dataloader, device, pbar, is_d3s=False
    )

    dataset = D3S(Path(FLAGS.d3s_root), split="val", transform=transform)
    num_classes = len(dataset.classes)
    num_bgs = len(dataset.backgrounds)

    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    d3s_top1, d3s_top5, d3s_stats = test(
        model,
        dataloader,
        device,
        pbar,
        is_d3s=True,
        num_classes=num_classes,
        num_bgs=num_bgs,
    )

    save_name = model_name.replace("/", "_")

    torch.save(
        {
            "imagenet_top1": imagenet_top1,
            "imagenet_top5": imagenet_top5,
            "d3s_top1": d3s_top1,
            "d3s_top5": d3s_top5,
            "d3s_stats": d3s_stats,
        },
        output_folder / f"{save_name}.pt",
    )


def main(argv):
    output_folder = Path(FLAGS.output_folder)

    model_names = [
        "resnet50",
        "resnext50_32x4d",
        "densenet121",
        "efficientnet_b0",
        "efficientnet_b5",
        # https://arxiv.org/abs/2010.11929
        "vit_base_patch16_224",
        "vit_large_patch32_384",
        "vit_base_patch16_224_sam",
        "vit_base_patch16_224_miil",
        "inception_resnet_v2",
        "mobilenetv3_small_100",
        "mobilenetv2_140",
        # https://arxiv.org/abs/1912.11370
        # https://github.com/google-research/big_transfer
        "resnetv2_101x3_bitm",  # pretrained on ImageNet21k, finetuned on ImageNet-1k
        "resnetv2_50x1_bit_distilled",  # pretrained on ImageNet-21k
        # https://arxiv.org/abs/1905.00546
        # https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
        "ssl_resnet50",  # pretrained on YFCC100M, finetuned on ImageNet-1k
        "ssl_resnext50_32x4d",  # pretrained on YFCC100M, finetuned on ImageNet-1k
        "swsl_resnet50",  # pretrained on images weakly matching ImageNet-1K synsets, finetuned on ImageNet-1K
        "swsl_resnext50_32x4d",  # pretrained on images weakly matching ImageNet-1K synsets, finetuned on ImageNet-1K
        # https://arxiv.org/pdf/1804.00097.pdf
        "adv_inception_v3",  # pretrained on ImageNet-1k, finetuned on ImageNet-1k
        "ens_adv_inception_resnet_v2",  # pretrained on ImageNet-1k, finetuned on ImageNet-1k
        # https://arxiv.org/abs/2104.14294
        # https://github.com/facebookresearch/dino
        "dino_vitb8",
        "dino_resnet50",
        # https://arxiv.org/abs/2106.08254
        "beit_base_patch16_224",
        "beit_large_patch16_512",
        "beitv2_large_patch16_224",
        # https://github.com/facebookresearch/deit
        # https://arxiv.org/abs/2012.12877,
        "deit_base_patch16_384",
        "deit3_huge_patch14_224",
        "deit3_huge_patch14_224_in21ft1k",  # pretrained on ImageNet21k, finetuned on ImageNet-1k
        # https://arxiv.org/abs/2103.00020
        # https://github.com/openai/CLIP
        "clip-RN50",
        "clip-RN50x64",
        "clip-ViT-B/32",
        "clip-ViT-L/14@336px",
        # https://github.com/MadryLab/robustness
        "robust_resnet50_l2",
        "robust_resnet50_linf",
    ]

    total = len(FLAGS.model_names) * (
        len(ImageNet(split="val")) + len(D3S(Path(FLAGS.d3s_root), split="val"))
    )

    with tqdm(total=total) as pbar:
        for model_name in FLAGS.model_names:
            assert model_name in model_names, f"Unknown model name: {model_name}"
            pbar.set_description(f"Evaluating {model_name} on GPU {FLAGS.rank}")
            evaluate_model(model_name, output_folder, pbar)


if __name__ == "__main__":
    app.run(main)
