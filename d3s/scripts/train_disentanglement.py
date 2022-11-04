import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from absl import app, flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from d3s.datasets import D3S, ImageNet
from d3s.disentangled_model import DisentangledModel
from d3s.mine_loss import MINELoss

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_bool("only_disentangle", True, "Only train disentanglement layer")
flags.DEFINE_bool("lu_decompose", True, "Use LU decomposed linear layer")
flags.DEFINE_integer("num_iters", 100000, "Number of iterations to train for")
flags.DEFINE_integer("num_workers", 4, "Number of workers to use for data loading")
flags.DEFINE_integer("train_batch_size", 32, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("lr", 1e-3, "Learning rate")
flags.DEFINE_float("weight_decay", 1e-4, "Momentum for SGD")
flags.DEFINE_float("mine_ma_rate", 0.1, "Moving average rate for MINE")
flags.DEFINE_float("alpha", 0.1, "Weight for cross entropy loss")
flags.DEFINE_integer(
    "switch_frequency", 10, "How often to switch between optimizing model and mine"
)
flags.DEFINE_integer("test_every", 5000, "Test every n iterations")
flags.DEFINE_string("log_folder", None, "Path to log folder")


class Trainer:
    def __init__(self, model, alpha=1.0, only_disentangle=True) -> None:
        self.model = model
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mine_criterions = {
            "fg": MINELoss(
                feature_dim=self.model.disentangle.num_features // 2,
                ma_rate=FLAGS.mine_ma_rate,
            ).cuda(),
            "bg": MINELoss(
                feature_dim=self.model.disentangle.num_features // 2,
                ma_rate=FLAGS.mine_ma_rate,
            ).cuda(),
        }
        self.only_disentangle = only_disentangle
        if self.only_disentangle:
            self.model_optimizer = optim.SGD(
                self.model.disentangle.parameters(),
                lr=FLAGS.lr,
                weight_decay=FLAGS.weight_decay,
            )
            for name, parameter in self.model.named_parameters():
                if "disentangle" not in name:
                    parameter.requires_grad = False
        else:
            self.model_optimizer = optim.SGD(
                self.model.parameters(),
                lr=FLAGS.lr,
                weight_decay=FLAGS.weight_decay,
            )
        self._mine_optimizers = None
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer,
            patience=500,
            threshold=1e-3,
            cooldown=100,
            min_lr=1e-4,
        )
        self.alpha = alpha

    @property
    def mine_optimizers(self):
        if self._mine_optimizers is None:
            self._mine_optimizers = {
                "fg": optim.Adam(
                    self.mine_criterions["fg"].parameters(),
                    lr=FLAGS.lr,
                ),
                "bg": optim.Adam(
                    self.mine_criterions["bg"].parameters(),
                    lr=FLAGS.lr,
                ),
            }
        return self._mine_optimizers

    def reset_mine_optimizers(self):
        self._mine_optimizers = None

    def train(self, batch, optimize_mine):
        if self.only_disentangle:
            self.model.eval()
        else:
            self.model.train()

        _, fg_features, bg_features = self.model(batch["d3s_images"])

        if optimize_mine:
            self.mine_optimizers["fg"].zero_grad()
            loss = self.mine_criterions["fg"](
                fg_features, batch["d3s_bg_labels"], optimize_T=True
            )
            loss.backward(retain_graph=True)
            self.mine_optimizers["fg"].step()

            self.mine_optimizers["bg"].zero_grad()
            loss = self.mine_criterions["bg"](
                bg_features, batch["d3s_fg_labels"], optimize_T=True
            )
            loss.backward()
            self.mine_optimizers["bg"].step()
        else:
            self.model_optimizer.zero_grad()
            self.reset_mine_optimizers()

            fg_mine_loss = self.mine_criterions["fg"](
                fg_features, batch["d3s_bg_labels"], optimize_T=False
            )
            bg_mine_loss = self.mine_criterions["bg"](
                bg_features, batch["d3s_fg_labels"], optimize_T=False
            )
            mine_loss = -(fg_mine_loss + bg_mine_loss)

            if not self.only_disentangle:
                outputs, _, _ = self.model(batch["imagenet_images"])
                ce_loss = self.ce_criterion(outputs, batch["imagenet_labels"])
                total_loss = mine_loss + self.alpha * ce_loss
            else:
                total_loss = mine_loss

            total_loss.backward()
            self.model_optimizer.step()
            self.scheduler.step(total_loss)

            if self.only_disentangle:
                return total_loss.item()
            else:
                return total_loss.item(), ce_loss.item(), mine_loss.item()


@torch.no_grad()
def test(model, dataloader, desc):
    model.eval()
    top1 = top5 = total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, file=sys.stdout):
        images, labels = batch[:2]
        images, labels = images.cuda(), labels.cuda()
        outputs, _, _ = model(images)
        _, top5_indices = outputs.topk(5, dim=1)
        top1_indices = top5_indices[:, 0]
        top5 += (top5_indices == labels.unsqueeze(1)).sum().item()
        top1 += (top1_indices == labels).sum().item()
        total += labels.shape[0]
    return 100 * top1 / total, 100 * top5 / total


def main(argv):
    model = DisentangledModel(
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        lu_decompose=FLAGS.lu_decompose,
    )
    model.cuda()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize]
    )
    val_transform = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
    )

    if not FLAGS.only_disentangle:
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

    if not FLAGS.only_disentangle:
        train_imagenet_dataloader = DataLoader(
            train_imagenet,
            batch_size=FLAGS.train_batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
        )
        train_imagenet_iter = iter(train_imagenet_dataloader)
    val_imagenet_dataloader = DataLoader(
        val_imagenet,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    train_d3s_dataloader = DataLoader(
        train_d3s,
        batch_size=FLAGS.train_batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    train_d3s_iter = iter(train_d3s_dataloader)
    val_d3s_dataloader = DataLoader(
        val_d3s,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    trainer = Trainer(model, FLAGS.alpha, only_disentangle=FLAGS.only_disentangle)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")

    writer = SummaryWriter(log_dir=log_folder)

    optimize_mine = True

    for i in trange(1, FLAGS.num_iters + 1):
        try:
            d3s_batch = next(train_d3s_iter)
        except StopIteration:
            train_d3s_iter = iter(train_d3s_dataloader)
            d3s_batch = next(train_d3s_iter)

        batch = {
            "d3s_images": d3s_batch[0].cuda(),
            "d3s_fg_labels": d3s_batch[1].cuda(),
            "d3s_bg_labels": d3s_batch[2].cuda(),
        }

        if not FLAGS.only_disentangle:
            try:
                imagenet_batch = next(train_imagenet_iter)
            except StopIteration:
                train_imagenet_iter = iter(train_imagenet_dataloader)
                imagenet_batch = next(train_imagenet_iter)

            batch["imagenet_images"] = imagenet_batch[0].cuda()
            batch["imagenet_labels"] = imagenet_batch[1].cuda()

        if i % FLAGS.switch_frequency == 0:
            optimize_mine = not optimize_mine
        if FLAGS.only_disentangle:
            total_loss = trainer.train(batch, optimize_mine=optimize_mine)
        else:
            total_loss, ce_loss, mine_loss = trainer.train(
                batch, optimize_mine=optimize_mine
            )
            writer.add_scalar("loss/ce_loss", ce_loss, i)
            writer.add_scalar("loss/mine_loss", mine_loss, i)
        if not optimize_mine:
            writer.add_scalar("loss/total_loss", total_loss, i)

        if i % FLAGS.test_every == 0:
            if not FLAGS.only_disentangle:
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
                    "optimizer": trainer.model_optimizer.state_dict(),
                },
                log_folder / f"ckpt-{i}.pth",
            )

    if not FLAGS.only_disentangle:
        writer.add_custom_scalars(
            {
                "imagenet": {
                    "imagenet": ["Multiline", ["imagenet/top1", "imagenet/top5"]]
                },
                "d3s": {"d3s": ["Multiline", ["d3s/top1", "d3s/top5"]]},
            }
        )
    writer.close()


if __name__ == "__main__":
    flags.mark_flags_as_required(["log_folder"])
    app.run(main)
