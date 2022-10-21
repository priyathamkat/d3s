import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from absl import app, flags
from d3s.datasets import D3S, ImageNet
from d3s.ranked_info_nce import RankedInfoNCE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_integer("num_iters", 100, "Number of iterations to train for")
flags.DEFINE_integer("train_batch_size", 4, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("lr", 5e-1, "Learning rate")
flags.DEFINE_float("momentum", 0.9, "Momentum for SGD")
flags.DEFINE_float("t", 1.0, "Temperature for InfoNCE")
flags.DEFINE_float("alpha_sc", 1.0, "Weight for same class images")
flags.DEFINE_float("alpha_sb", 1.0, "Weight for same background images")
flags.DEFINE_integer("test_every", 100, "Test every n iterations")
flags.DEFINE_string(
    "log_folder", "/cmlscratch/pkattaki/void/d3s/d3s/logs", "Path to log folder"
)


class Trainer:
    def __init__(self, model, t, alphas) -> None:
        self.model = model
        self.ce_criterion = nn.CrossEntropyLoss()
        self.rince_criterion = RankedInfoNCE(t=t)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=True
        )
        self.alphas = alphas

    def train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch["imagenet_images"])
        loss = self.ce_criterion(outputs, batch["imagenet_labels"])
        for quadruplet in batch["d3s_images"]:
            query = self.model(quadruplet["query"])
            same_class = self.model(quadruplet["same_class"])
            same_bg = self.model(quadruplet["same_bg"])
            negatives = self.model(quadruplet["negatives"])
            results = torch.stack([negatives, same_bg, same_class], dim=1)
            loss += self.rince_criterion(query, results, self.alphas)
        loss.backward()
        self.optimizer.step()
        return loss.item()


@torch.no_grad()
def test(model, dataloader, desc):
    model.eval()
    top1 = top5 = total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, file=sys.stdout):
        images, labels = batch[:2]
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, top5_indices = outputs.topk(5, dim=1)
        top1_indices = top5_indices[:, 0]
        top5 += (top5_indices == labels.unsqueeze(1)).sum().item()
        top1 += (top1_indices == labels).sum().item()
        total += labels.shape[0]
    return 100 * top1 / total, 100 * top5 / total


def main(argv):
    model = models.resnet50(pretrained=True)
    model.cuda()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize]
    )
    val_transform = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
    )

    train_imagenet = ImageNet(split="train", transform=train_transform)
    val_imagenet = ImageNet(split="val", transform=val_transform)
    train_d3s = D3S(
        Path(FLAGS.d3s_root),
        split="train",
        shift="background-shift",
        transform=train_transform,
    )
    val_d3s = D3S(
        Path(FLAGS.d3s_root),
        split="val",
        shift="background-shift",
        transform=val_transform,
    )

    train_imagenet_dataloader = DataLoader(
        train_imagenet,
        batch_size=FLAGS.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_imagenet_iter = iter(train_imagenet_dataloader)
    val_imagenet_dataloader = DataLoader(
        val_imagenet,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_d3s_dataloader = DataLoader(
        val_d3s,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    alphas = torch.tensor([FLAGS.alpha_sb, FLAGS.alpha_sc], device="cuda")
    trainer = Trainer(model, FLAGS.t, alphas)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")
    
    writer = SummaryWriter(log_dir=log_folder)

    for i in trange(1, FLAGS.num_iters + 1):
        try:
            images, labels = next(train_imagenet_iter)
        except StopIteration:
            train_imagenet_iter = iter(train_imagenet_dataloader)
            images, labels = next(train_imagenet_iter)

        batch = {"imagenet_images": images.cuda(), "imagenet_labels": labels.cuda()}
        batch["d3s_images"] = []

        for label in labels:
            label = label.item()
            query, _, bg_idx = zip(*train_d3s.get_random(class_idx=label))
            query, bg_idx = query[0], bg_idx[0]
            query = query.unsqueeze(0)
            same_class, _, _ = zip(
                *train_d3s.get_random(
                    class_idx=label, num_samples=FLAGS.train_batch_size
                )
            )
            same_bg, _, _ = zip(
                *train_d3s.get_random(
                    bg_idx=bg_idx,
                    not_class_idx=label,
                    num_samples=FLAGS.train_batch_size,
                )
            )
            negatives, _, _ = zip(
                *train_d3s.get_random(
                    not_class_idx=label,
                    not_bg_idx=bg_idx,
                    num_samples=FLAGS.train_batch_size,
                )
            )

            batch["d3s_images"].append(
                {
                    "query": query.cuda(),
                    "same_class": torch.stack(same_class).cuda(),
                    "same_bg": torch.stack(same_bg).cuda(),
                    "negatives": torch.stack(negatives).cuda(),
                }
            )

        loss = trainer.train(batch)
        writer.add_scalar("loss", loss, i)

        if i % FLAGS.test_every == 0:
            top1, top5 = test(
                model,
                val_imagenet_dataloader,
                f"Testing on ImageNet after {i} iterations",
            )
            writer.add_scalar("imagenet/top1", top1, i)
            writer.add_scalar("imagenet/top5", top5, i)
            top1, top5 = test(
                model, val_d3s_dataloader, f"Testing on D3S after {i} iterations"
            )
            writer.add_scalar("d3s/top1", top1, i)
            writer.add_scalar("d3s/top5", top5, i)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                },
                log_folder / f"ckpt-{i}.pth",
            )

    writer.add_custom_scalars(
        {
            "imagenet": {"imagenet": ["Multiline", ["imagenet/top1", "imagenet/top5"]]},
            "d3s": {"d3s": ["Multiline", ["d3s/top1", "d3s/top5"]]},
        }
    )
    writer.close()


if __name__ == "__main__":
    app.run(main)
