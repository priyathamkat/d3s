import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from absl import app, flags
from d3s.contrastive_loss import ContrastiveLoss
from d3s.datasets import D3S, ImageNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

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
flags.DEFINE_integer("train_batch_size", 4, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("lr", 1e-2, "Learning rate")
flags.DEFINE_float("momentum", 0.9, "Momentum for SGD")
flags.DEFINE_float("weight_decay", 1e-4, "Momentum for SGD")
flags.DEFINE_float("t", 0.3, "Temperature for InfoNCE")
flags.DEFINE_float("alpha", 0.1, "Weight for cross entropy loss")
flags.DEFINE_integer("test_every", 5000, "Test every n iterations")
flags.DEFINE_string(
    "log_folder", "/cmlscratch/pkattaki/void/d3s/d3s/logs", "Path to log folder"
)


class InvertibleLinear(nn.Module):
    def __init__(self, num_features, lu_decompose=True, bias=False):
        super().__init__()
        self.num_features = num_features
        self.lu_decompose = lu_decompose

        if self.lu_decompose:
            w_init, _ = torch.linalg.qr(
                torch.randn(self.num_features, self.num_features)
            )
            p, lower, upper = torch.linalg.lu(w_init)
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, diagonal=1)
            l_mask = torch.tril(torch.ones_like(w_init), diagonal=-1)
            eye = torch.eye(self.num_features)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("eye", eye)
        else:
            weight = torch.empty(self.num_features, self.num_features)
            nn.init.xavier_normal_(weight)
            self.weight = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.zeros(self.num_features)) if bias else None

    def forward(self, x):
        if self.lu_decompose:
            lower = self.lower * self.l_mask + self.eye
            upper = self.upper * self.l_mask.t().contiguous()
            upper = upper + torch.diag(self.sign_s * torch.exp(self.log_s))
            weight = self.p @ lower @ upper
            features = F.linear(x, weight, self.bias)
        else:
            features = F.linear(x, self.weight, self.bias)
        return (
            features[:, : self.num_features // 2],
            features[:, self.num_features // 2 :],
        )


class FeatureModel(nn.Module):
    def __init__(self, model, lu_decompose=True):
        super().__init__()
        self.model = model
        self.fc = self.model.fc
        self.disentangle = InvertibleLinear(
            self.fc.in_features, lu_decompose=lu_decompose
        )
        self.model.fc = nn.Identity()

    def forward(self, x):
        features = self.model(x)
        fg_features, bg_features = self.disentangle(features)
        outputs = self.fc(features)
        return outputs, fg_features, bg_features


class Trainer:
    def __init__(self, model, t, alpha, only_disentangle=True) -> None:
        self.model = model
        self.ce_criterion = nn.CrossEntropyLoss()
        self.contrasive_criterion = ContrastiveLoss(t=t)
        self.only_disentangle = only_disentangle
        if self.only_disentangle:
            self.optimizer = optim.SGD(
                self.model.disentangle.parameters(),
                lr=FLAGS.lr,
                momentum=FLAGS.momentum,
                weight_decay=FLAGS.weight_decay,
            )
            for name, parameter in self.model.named_parameters():
                if "disentangle" not in name:
                    parameter.requires_grad = False
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=FLAGS.lr,
                momentum=FLAGS.momentum,
                weight_decay=FLAGS.weight_decay,
            )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, threshold=1e-3, cooldown=50, min_lr=1e-4
        )
        self.alpha = alpha

    def train(self, batch):
        if self.only_disentangle:
            self.model.eval()
        else:
            self.model.train()
        self.optimizer.zero_grad()
        ce_loss = None
        if not self.only_disentangle:
            outputs, _, _ = self.model(batch["imagenet_images"])
            ce_loss = self.ce_criterion(outputs, batch["imagenet_labels"])
        contrastive_loss = 0
        for quadruplet in batch["d3s_images"]:
            _, query_fg, query_bg = self.model(quadruplet["query"])
            _, same_class_fg, _ = self.model(quadruplet["same_class"])
            _, same_bg_fg, same_bg_bg = self.model(quadruplet["same_bg"])
            _, negatives_fg, negatives_bg = self.model(quadruplet["negatives"])
            contrastive_loss += self.contrasive_criterion(
                query_fg, same_class_fg, torch.concat([negatives_fg, same_bg_fg])
            )
            contrastive_loss += self.contrasive_criterion(
                query_bg, same_bg_bg[0:], negatives_bg
            )
        contrastive_loss /= len(batch["d3s_images"])
        if self.only_disentangle:
            total_loss = contrastive_loss
        else:
            total_loss = contrastive_loss + self.alpha * ce_loss
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step(total_loss)
        if self.only_disentangle:
            return total_loss.item()
        else:
            return total_loss.item(), ce_loss.item(), contrastive_loss.item()


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
    model = FeatureModel(
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
    val_d3s_dataloader = DataLoader(
        val_d3s,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    trainer = Trainer(
        model, FLAGS.t, FLAGS.alpha, only_disentangle=FLAGS.only_disentangle
    )

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")

    writer = SummaryWriter(log_dir=log_folder)

    for i in trange(1, FLAGS.num_iters + 1):
        if FLAGS.only_disentangle:
            batch = {}
            labels = torch.randint(0, 1000, (FLAGS.train_batch_size,))
        else:
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
                *train_d3s.get_random(class_idx=label, num_samples=1)
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

        if FLAGS.only_disentangle:
            total_loss = trainer.train(batch)
        else:
            total_loss, ce_loss, contrastive_loss = trainer.train(batch)
            writer.add_scalar("loss/ce_loss", ce_loss, i)
            writer.add_scalar("loss/contrastive_loss", contrastive_loss, i)
        writer.add_scalar("loss/total_loss", total_loss, i)

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
