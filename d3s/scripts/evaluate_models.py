from pathlib import Path

import timm
import torch
import torch.multiprocessing as mp
from absl import app, flags
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3s.datasets import D3S, ImageNet

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_integer("batch_size", 128, "Batch size for evaluating the model")
flags.DEFINE_string(
    "output_folder",
    "/cmlscratch/pkattaki/void/d3s/d3s/outputs/d3s_eval_stats",
    "Folder to save the outputs",
)
flags.DEFINE_integer("num_gpus", 8, "Number of GPUs to use")
flags.DEFINE_integer("num_workers", 8, "Number of workers to use")


@torch.no_grad()
def test(model, dataloader, device, is_d3s=False, num_classes=0, num_bgs=0):
    model.eval()
    top1 = top5 = total = 0
    # stats: {top1, top5, total} X classes X backgrounds
    stats = torch.zeros(3, num_classes, num_bgs) if is_d3s else None
    for batch in dataloader:
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        images, class_idxs = batch[:2]

        outputs = model(images)
        _, top5_preds = outputs.topk(5, dim=1)
        top1_preds = top5_preds[:, 0]

        is_correct_top1 = (top1_preds == class_idxs).cpu().float()
        is_correct_top5 = (top5_preds == class_idxs.unsqueeze(1)).any(dim=1).cpu().float()

        if is_d3s:
            bg_idxs = batch[2]

            stats[0, class_idxs, bg_idxs] += is_correct_top1
            stats[1, class_idxs, bg_idxs] += is_correct_top5
            stats[2, class_idxs, bg_idxs] += 1

        top5 += is_correct_top5.sum().item()
        top1 += is_correct_top1.sum().item()
        total += class_idxs.shape[0]

    return 100 * top1 / total, 100 * top5 / total, stats


def evaluate_model(rank, queue, output_folder):
    device = torch.device(f"cuda:{rank}")

    while True:
        model_name = queue.get()

        if model_name is None:
            break

        model = timm.create_model(model_name, pretrained=True).to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        dataset = ImageNet(split="val", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
        )
        imagenet_top1, imagenet_top5, _ = test(model, dataloader, device, is_d3s=False)

        dataset = D3S(
            Path(FLAGS.d3s_root), split="val", transform=transform
        )
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
            is_d3s=True,
            num_classes=num_classes,
            num_bgs=num_bgs,
        )

        print(imagenet_top1, imagenet_top5, d3s_top1, d3s_top5)

        torch.save(
            {
                "imagenet_top1": imagenet_top1,
                "imagenet_top5": imagenet_top5,
                "d3s_top1": d3s_top1,
                "d3s_top5": d3s_top5,
                "d3s_stats": d3s_stats,
            },
            output_folder / f"{model_name}.pt",
        )


def main(argv):
    torch.use_deterministic_algorithms(True)
    output_folder = Path(FLAGS.output_folder)
    queue = mp.Queue(maxsize=2 * FLAGS.num_gpus)
    processes = []
    for rank in range(FLAGS.num_gpus):
        p = mp.Process(target=evaluate_model, args=(rank, queue, output_folder))
        p.start()
        processes.append(p)

    model_names = [
        "resnet50",
        "resnext50_32x4d",
        "densenet121",
        "efficientnet_b1",
        "vit_base_patch16_224",
        "inception_resnet_v2",
        "mobilenetv3_small_100",
    ]

    for model_name in tqdm(model_names):
        queue.put(model_name)

    for _ in range(FLAGS.num_gpus):
        queue.put(None)  # sentinel values to signal subprocesses to exit

    for p in processes:
        p.join()  # wait for all subprocesses to finish


if __name__ == "__main__":
    app.run(main)
