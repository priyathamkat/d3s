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
flags.DEFINE_integer("num_bg_features", 250, "Number of background features")
flags.DEFINE_integer("num_iters", 100000, "Number of iterations to train for")
flags.DEFINE_integer("num_workers", 4, "Number of workers to use for data loading")
flags.DEFINE_integer("train_batch_size", 32, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("alpha", 1.0, "Weight for cross entropy loss")
flags.DEFINE_integer("train_model_every", 5, "Train model every n iterations")
flags.DEFINE_float("pretraining_lr", 1e-3, "Learning rate for pretraining")
flags.DEFINE_float("model_lr", 5e-3, "Learning rate for model")
flags.DEFINE_float("discriminator_lr", 5e-3, "Learning rate for discriminators")
flags.DEFINE_integer("t_max", 100, "T_max for cosine annealing")
flags.DEFINE_float("eta_min", 5e-4, "eta_min for cosine annealing")
flags.DEFINE_float(
    "label_vector_dim_fraction",
    0.2,
    "Size of label vector as a fraction of feature_dim",
)
flags.DEFINE_integer("test_every", 5000, "Test every n iterations")
flags.DEFINE_string("log_folder", None, "Path to log folder")
flags.DEFINE_integer("num_pretraining_iters", 2000, "Number of pretraining iterations")


class WassersteinTrainer(nn.Module):
    def __init__(
        self,
        model,
        fg_classes=1000,
        bg_classes=10,
        alpha=1.0,
        lamb=10,
        train_model_every=5,
        pretraining_lr=1e-3,
        model_lr=1e-4,
        discriminator_lr=1e-4,
        T_max=100,
        eta_min=1e-4,
        label_vector_dim_fraction=0.2,
        num_pretraining_iters=2000,
    ) -> None:
        super().__init__()

        self.model = model
        self.ce_criterion = nn.CrossEntropyLoss()

        fg_label_vector_dim = int(
            self.model.num_fg_features * label_vector_dim_fraction
        )
        bg_label_vector_dim = int(
            self.model.num_bg_features * label_vector_dim_fraction
        )

        self.register_buffer(
            "fg_label_matrix", torch.randn(bg_classes, fg_label_vector_dim)
        )
        self.register_buffer(
            "bg_label_matrix", torch.randn(fg_classes, bg_label_vector_dim)
        )

        self.fg_discriminator = self.create_mlp_discriminator(
            self.model.num_fg_features + fg_label_vector_dim
        )
        self.bg_discriminator = self.create_mlp_discriminator(
            self.model.num_bg_features + bg_label_vector_dim
        )

        self.pretraining_optimizer = optim.SGD(
            self.model.parameters(), lr=pretraining_lr, momentum=0.9
        )
        self.model_optimizer = self.create_optimizer(self.model, model_lr)
        self.discriminator_optimizers = {
            "fg": self.create_optimizer(self.fg_discriminator, discriminator_lr),
            "bg": self.create_optimizer(self.bg_discriminator, discriminator_lr),
        }

        self.discriminator_schedulers = {
            k: self.create_scheduler(v, T_max, eta_min)
            for k, v in self.discriminator_optimizers.items()
        }
        self.model_scheduler = self.create_scheduler(
            self.model_optimizer, T_max, eta_min
        )

        self.alpha = alpha
        self.lamb = lamb

        self.pretraining = True
        self.train_model_every = train_model_every
        self.train_discriminators = True
        self._train_count = 0
        self.num_pretraining_iters = num_pretraining_iters

    def create_mlp_discriminator(self, feature_dim):
        layers = []
        while feature_dim != 1:
            layers.extend(
                [
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.LeakyReLU(negative_slope=0.1),
                ]
            )
            feature_dim //= 2
        mlp = nn.Sequential(*layers[:-1])
        return mlp

    def create_optimizer(self, model, lr):
        return optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))

    def create_scheduler(self, optimizer, T_max, eta_min):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    def cat_features_labels(self, features, label_vectors):
        return torch.cat((features, label_vectors), dim=1)

    def compute_discriminator(self, features, labels, discriminator, label_matrix):
        label_vectors = label_matrix[labels]
        x_f_l = self.cat_features_labels(features, label_vectors)
        return discriminator(x_f_l)

    def shuffle(self, x):
        return x[torch.randperm(x.shape[0])]

    def compute_discriminator_loss(self, features, labels, discriminator, label_matrix):
        label_vectors = label_matrix[labels]
        x_f_l = self.cat_features_labels(features, label_vectors)
        d_f_l = discriminator(x_f_l)
        shuffled_label_vectors = self.shuffle(label_vectors)
        x_f_shuffled_l = self.cat_features_labels(features, shuffled_label_vectors)
        d_f_shuffled_l = discriminator(x_f_shuffled_l)
        epsilons = torch.rand_like(x_f_l)
        x_f_interpolated_l = epsilons * x_f_l + (1 - epsilons) * x_f_shuffled_l
        d_f_interpolated_l = discriminator(x_f_interpolated_l)
        gradient = torch.autograd.grad(
            d_f_interpolated_l,
            x_f_interpolated_l,
            grad_outputs=torch.ones_like(d_f_interpolated_l),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = (gradient.norm(2, dim=1) - 1) ** 2
        return (d_f_l - d_f_shuffled_l + self.lamb * gradient_penalty).mean()

    def train(self, batch):
        self.model.train()

        if self._train_count == self.num_pretraining_iters:
            self.pretraining = False
        self._train_count += 1

        fg_outputs, bg_outputs, fg_features, bg_features = self.model(batch["images"])
        num_d3s_samples = batch["bg_labels"].shape[0]
        if self.train_discriminators and not self.pretraining:

            self.discriminator_optimizers["fg"].zero_grad()
            fg_discriminator_loss = self.compute_discriminator_loss(
                fg_features[:num_d3s_samples],
                batch["bg_labels"],
                self.fg_discriminator,
                self.fg_label_matrix,
            )
            fg_discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizers["fg"].step()

            self.discriminator_optimizers["bg"].zero_grad()
            bg_discriminator_loss = self.compute_discriminator_loss(
                bg_features,
                batch["fg_labels"],
                self.bg_discriminator,
                self.bg_label_matrix,
            )
            bg_discriminator_loss.backward()
            self.discriminator_optimizers["bg"].step()

            for scheduler in self.discriminator_schedulers.values():
                scheduler.step()

            if self._train_count % self.train_model_every == 0:
                self.train_discriminators = False

            return fg_discriminator_loss.item(), bg_discriminator_loss.item()
        else:
            self.model_optimizer.zero_grad()

            if not self.pretraining:
                d_fg_bg = self.compute_discriminator(
                    fg_features[:num_d3s_samples],
                    batch["bg_labels"],
                    self.fg_discriminator,
                    self.fg_label_matrix,
                )
                d_fg_shuffled_bg = self.compute_discriminator(
                    fg_features[:num_d3s_samples],
                    self.shuffle(batch["bg_labels"]),
                    self.fg_discriminator,
                    self.fg_label_matrix,
                )

                fg_loss = (d_fg_shuffled_bg - d_fg_bg).mean()
                fg_loss.backward(retain_graph=True)

                d_bg_fg = self.compute_discriminator(
                    bg_features,
                    batch["fg_labels"],
                    self.bg_discriminator,
                    self.bg_label_matrix,
                )
                d_bg_shuffled_fg = self.compute_discriminator(
                    bg_features,
                    self.shuffle(batch["fg_labels"]),
                    self.bg_discriminator,
                    self.bg_label_matrix,
                )

                bg_loss = (d_bg_shuffled_fg - d_bg_fg).mean()
                bg_loss.backward(retain_graph=True)
            else:
                fg_loss = torch.tensor(0.0)
                bg_loss = torch.tensor(0.0)

            ce_loss = self.ce_criterion(fg_outputs, batch["fg_labels"])
            ce_loss += self.ce_criterion(
                bg_outputs[:num_d3s_samples], batch["bg_labels"]
            )
            ce_loss *= self.alpha
            ce_loss.backward()

            if self.pretraining:
                self.pretraining_optimizer.step()
            else:
                self.model_optimizer.step()
                self.model_scheduler.step()

            self.train_discriminators = True

            return fg_loss.item(), bg_loss.item(), ce_loss.item() / self.alpha


@torch.no_grad()
def test(model, dataloader, desc):
    model.eval()
    fg_top1 = fg_top5 = bg_top1 = total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, file=sys.stdout):
        images, labels = batch[:2]
        images, labels = images.cuda(), labels.cuda()
        fg_outputs, bg_outputs, _, _ = model(images)
        _, fg_top5_indices = fg_outputs.topk(5, dim=1)
        fg_top1_indices = fg_top5_indices[:, 0]
        fg_top5 += (fg_top5_indices == labels.unsqueeze(1)).sum().item()
        fg_top1 += (fg_top1_indices == labels).sum().item()
        if len(batch) == 3:
            bg_top1 += (bg_outputs.argmax(dim=1).cpu() == batch[2]).sum().item()
        total += labels.shape[0]
    return 100 * fg_top1 / total, 100 * fg_top5 / total, 100 * bg_top1 / total


def main(argv):
    torch.backends.cudnn.benchmark = True
    model = DisentangledModel(
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        num_bg_features=FLAGS.num_bg_features,
    )
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

    trainer = WassersteinTrainer(
        model,
        alpha=FLAGS.alpha,
        train_model_every=FLAGS.train_model_every,
        model_lr=FLAGS.model_lr,
        discriminator_lr=FLAGS.discriminator_lr,
        T_max=FLAGS.t_max,
        eta_min=FLAGS.eta_min,
        label_vector_dim_fraction=FLAGS.label_vector_dim_fraction,
        num_pretraining_iters=FLAGS.num_pretraining_iters,
    ).cuda()

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")

    writer = SummaryWriter(log_dir=log_folder)

    for i in trange(1, FLAGS.num_iters + 1):
        try:
            d3s_batch = next(train_d3s_iter)
        except StopIteration:
            train_d3s_iter = iter(train_d3s_dataloader)
            d3s_batch = next(train_d3s_iter)

        try:
            imagenet_batch = next(train_imagenet_iter)
        except StopIteration:
            train_imagenet_iter = iter(train_imagenet_dataloader)
            imagenet_batch = next(train_imagenet_iter)

        batch = {
            "images": torch.cat([d3s_batch[0], imagenet_batch[0]]).cuda(),
            "fg_labels": torch.cat([d3s_batch[1], imagenet_batch[1]]).cuda(),
            "bg_labels": d3s_batch[2].cuda(),
        }

        losses = trainer.train(batch)

        if len(losses) == 2:
            writer.add_scalar("loss/fg_discriminator_loss", losses[0], i)
            writer.add_scalar("loss/bg_discriminator_loss", losses[1], i)
        else:
            writer.add_scalar("loss/fg_loss", losses[0], i)
            writer.add_scalar("loss/bg_loss", losses[1], i)
            writer.add_scalar("loss/ce_loss", losses[2], i)

        if i % FLAGS.test_every == 0:
            fg_top1, fg_top5, _ = test(
                model,
                val_imagenet_dataloader,
                f"Testing on ImageNet after {i} iterations",
            )
            writer.add_scalar("imagenet/fg_top1", fg_top1, i)
            writer.add_scalar("imagenet/fg_top5", fg_top5, i)
            fg_top1, fg_top5, bg_top1 = test(
                model, val_d3s_dataloader, f"Testing on D3S after {i} iterations"
            )
            writer.add_scalar("d3s/fg_top1", fg_top1, i)
            writer.add_scalar("d3s/fg_top5", fg_top5, i)
            writer.add_scalar("d3s/bg_top1", bg_top1, i)
            torch.save(
                {
                    "trainer": trainer.state_dict(),
                    "model_optimizer": trainer.model_optimizer.state_dict(),
                    "discriminator_optimizers": {
                        k: v.state_dict()
                        for k, v in trainer.discriminator_optimizers.items()
                    },
                    "model_scheduler": trainer.model_scheduler,
                    "discriminator_schedulers": trainer.discriminator_schedulers,
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
    flags.mark_flags_as_required(["log_folder"])
    app.run(main)
