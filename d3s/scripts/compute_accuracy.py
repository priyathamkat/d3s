import sys
from pathlib import Path

from absl import app, flags
from d3s.datasets import D3S
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.models as models
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "images_dir", None, "Directory containing images to compute diversity for"
)
flags.DEFINE_string("model", "resnet50", "Model to compute accuracy for")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluating the model")


def main(argv):
    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = D3S(Path(FLAGS.images_dir), transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    model = getattr(models, FLAGS.model)(pretrained=True).cuda()
    model.eval()
    top1 = top5 = total = 0
    for images, labels in tqdm(dataloader, file=sys.stdout):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, top5_indices = outputs.topk(5, dim=1)
        top1_indices = top5_indices[:, 0]
        top5 += (top5_indices == labels.unsqueeze(1)).sum().item()
        top1 += (top1_indices == labels).sum().item()
        total += labels.shape[0]

    print(f"Top-1 Accuracy of {FLAGS.model} on {FLAGS.images_dir}: {100 * top1 / total:.2f}")
    print(f"Top-5 Accuracy of {FLAGS.model} on {FLAGS.images_dir}: {100 * top5 / total:.2f}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["images_dir"])
    app.run(main)
