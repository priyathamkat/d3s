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
from d3s.disentangled_model_variable import DisentangledModelVariable

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_string("model_ckpt", None, "Path to model checkpoint")
flags.DEFINE_integer("num_iters", 100000, "Number of iterations to train for")
flags.DEFINE_integer("num_workers", 4, "Number of workers to use for data loading")
flags.DEFINE_integer("train_batch_size", 64, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("lr", 1e-3, "Learning rate")
flags.DEFINE_float("momentum", 0.9, "Momentum for SGD")
flags.DEFINE_float("weight_decay", 1e-4, "Momentum for SGD")
flags.DEFINE_integer("test_every", 5000, "Test every n iterations")
flags.DEFINE_string(
    "log_folder",
    "/fs/nexus-scratch/alevine0/d3s/d3s/logs/test_disentanglement",
    "Path to log folder",
)


class DisentanglementTestingModel(nn.Module):
    def __init__(self, model, fg_classes=1000, bg_classes=10):
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.fg_head = nn.Linear(self.model.disentangle.split, fg_classes)
        self.bg_head = nn.Linear(self.model.disentangle.num_features - self.model.disentangle.split, bg_classes)
        self.heads = nn.ModuleList([self.fg_head, self.bg_head])

    def forward(self, x):
        with torch.no_grad():
            _, fg_features, bg_features = self.model(x)
        fg_logits = self.fg_head(fg_features)
        bg_logits = self.bg_head(bg_features)
        return fg_logits, bg_logits


class Trainer:
    def __init__(self, model) -> None:
        self.model = model
        self.ce_criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.heads.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
        )

    def train(self, batch):
        self.optimizer.zero_grad()
        fg_logits, bg_logits = self.model(batch["images"])
        #fg_loss = self.ce_criterion(fg_logits, batch["fg_labels"])
        bg_labels = batch["bg_labels"]
        bg_loss = self.ce_criterion(bg_logits[-bg_labels.shape[0] :], bg_labels)
        total_loss =  bg_loss# fg_loss + bg_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), bg_loss.item()


@torch.no_grad()
def test(model, dataloader, desc):
    model.eval()
    fg_top1 = fg_top5 = bg_top1 = total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, file=sys.stdout):
        images = batch[0].cuda()
        fg_labels = batch[1].cuda()
        try:
            bg_labels = batch[2].cuda()
        except IndexError:
            bg_labels = None
        fg_outputs, bg_outputs = model(images)
        _, fg_top5_indices = fg_outputs.topk(5, dim=1)
        fg_top1_indices = fg_top5_indices[:, 0]
        fg_top5 += (fg_top5_indices == fg_labels.unsqueeze(1)).sum().item()
        fg_top1 += (fg_top1_indices == fg_labels).sum().item()
        if bg_labels is not None:
            bg_top1 += (bg_outputs.argmax(dim=1) == bg_labels).sum().item()
        total += fg_labels.shape[0]
    fg_top1 = 100 * fg_top1 / total
    fg_top5 = 100 * fg_top5 / total
    bg_top1 = 100 * bg_top1 / total
    return fg_top1, fg_top5, bg_top1


def main(argv):
    
    ckpt = torch.load(FLAGS.model_ckpt)
    fg_weights = ckpt['foreground'].float()
    bg_weights = ckpt['background'].float()

    disentangled_model = DisentangledModelVariable(
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),fg_weights,bg_weights)

    testing_model = DisentanglementTestingModel(disentangled_model)
    testing_model = testing_model.cuda()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose(
        #[T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize]
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
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
    # train_imagenet_dataloader = DataLoader(
    #     train_imagenet,
    #     batch_size=FLAGS.train_batch_size // 2,
    #     shuffle=True,
    #     num_workers=FLAGS.num_workers,
    #     pin_memory=True,
    # )
    # train_imagenet_iter = iter(train_imagenet_dataloader)
    # val_imagenet_dataloader = DataLoader(
    #     val_imagenet,
    #     batch_size=FLAGS.val_batch_size,
    #     shuffle=False,
    #     num_workers=FLAGS.num_workers,
    #     pin_memory=True,
    # )
    train_d3s_dataloader = DataLoader(
        train_d3s,
        batch_size=FLAGS.train_batch_size // 2,
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

    trainer = Trainer(testing_model)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")

    writer = SummaryWriter(log_dir=log_folder)

    for i in trange(1, FLAGS.num_iters + 1):
        # try:
        #     imagenet_batch = next(train_imagenet_iter)
        # except StopIteration:
        #     train_imagenet_iter = iter(train_imagenet_dataloader)
        #     imagenet_batch = next(train_imagenet_iter)

        try:
            d3s_batch = next(train_d3s_iter)
        except StopIteration:
            train_d3s_iter = iter(train_d3s_dataloader)
            d3s_batch = next(train_d3s_iter)

        batch = {
            "images": d3s_batch[0].cuda(), #torch.cat([imagenet_batch[0], d3s_batch[0]], dim=0).cuda(),
            "fg_labels": d3s_batch[1].cuda(), #torch.cat([imagenet_batch[1], d3s_batch[1]], dim=0).cuda(),
            "bg_labels": d3s_batch[2].cuda(),
        }

        total_loss,  bg_loss = trainer.train(batch)

        #writer.add_scalar("loss/total_loss", total_loss, i)
       # writer.add_scalar("loss/fg_loss", fg_loss, i)
        writer.add_scalar("loss/bg_loss", bg_loss, i)

        if i % FLAGS.test_every == 0:
            # fg_top1, fg_top5, _ = test(
            #     testing_model,
            #     val_imagenet_dataloader,
            #     f"Testing on ImageNet after {i} iterations",
            # )
            # writer.add_scalar("imagenet/fg_top1", fg_top1, i)
            # writer.add_scalar("imagenet/fg_top5", fg_top5, i)
            fg_top1, fg_top5, bg_top1 = test(
                testing_model,
                val_d3s_dataloader,
                f"Testing on D3S after {i} iterations",
            )
            writer.add_scalar("d3s/fg_top1", fg_top1, i)
            writer.add_scalar("d3s/fg_top5", fg_top5, i)
            writer.add_scalar("d3s/bg_top1", bg_top1, i)

    writer.add_custom_scalars(
        {
        #    "imagenet": {"imagenet": ["Multiline", ["imagenet/top1", "imagenet/top5"]]},
            "d3s": {"d3s": ["Multiline", ["d3s/top1", "d3s/top5"]]},
        }
    )
    writer.close()


if __name__ == "__main__":
    flags.mark_flags_as_required(["model_ckpt"])
    app.run(main)
