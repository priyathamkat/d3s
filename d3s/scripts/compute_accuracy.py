import sys
from pathlib import Path

import torch
import torchvision.models as models
from absl import app, flags
from d3s.datasets import D3S
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_enum(
    "shift",
    "all",
    ["all", "background-shift", "geography-shift", "time-shift"],
    "Shift to use for evaluation",
)
flags.DEFINE_string("model", "resnet50", "Model to compute accuracy for")
flags.DEFINE_integer("batch_size", 128, "Batch size for evaluating the model")


def main(argv):
    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = D3S(Path(FLAGS.d3s_root), shift=FLAGS.shift, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model = getattr(models, FLAGS.model)(pretrained=True).cuda()
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, file=sys.stdout):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, top5_indices = outputs.topk(5, dim=1)
            top1_indices = top5_indices[:, 0]
            top5 += (top5_indices == labels.unsqueeze(1)).sum().item()
            top1 += (top1_indices == labels).sum().item()
            total += labels.shape[0]

    print(
        f"Top-1 Accuracy of {FLAGS.model} on D3S ({FLAGS.shift}): {100 * top1 / total:.2f}"
    )
    print(
        f"Top-5 Accuracy of {FLAGS.model} on D3S ({FLAGS.shift}): {100 * top5 / total:.2f}"
    )


if __name__ == "__main__":
    app.run(main)
