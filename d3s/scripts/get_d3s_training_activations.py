from d3s.datasets import D3S
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = T.Compose(
	[T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
)
train_d3s = D3S(
	Path('/cmlscratch/pkattaki/datasets/d3s'),
	split="train",
	shift="background-shift",
	transform=transform,
)

train_dataloader = DataLoader(
	train_d3s,
	batch_size=50,
	shuffle=False,
	num_workers=4,
	pin_memory=True,
)

mod = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
mod.fc = nn.Identity()
mod = mod.cuda()
mod.eval()

train_classes = []
train_backgrounds = []
train_activations = []

for batch in tqdm(train_dataloader,   file=sys.stdout):
	with torch.no_grad():
		images, labels, backgrounds = batch
		images = images.cuda()
		outputs =  mod(images).cpu()
		train_classes.append(labels)
		train_backgrounds.append(backgrounds)
		train_activations.append(outputs)

torch.save({"classes": torch.stack(train_classes),
			"backgrounds": torch.stack(train_backgrounds),
			"activations": torch.stack(train_activations)}, "d3s_training_activations.pth")

val_d3s = D3S(
	Path('/cmlscratch/pkattaki/datasets/d3s'),
	split="val",
	shift="background-shift",
	transform=transform,
)

val_dataloader = DataLoader(
	val_d3s,
	batch_size=50,
	shuffle=False,
	num_workers=4,
	pin_memory=True,
)

val_classes = []
val_backgrounds = []
val_activations = []

for batch in tqdm(val_dataloader,   file=sys.stdout):
	with torch.no_grad():
		images, labels, backgrounds = batch
		images = images.cuda()
		outputs =  mod(images).cpu()
		val_classes.append(labels)
		val_backgrounds.append(backgrounds)
		val_activations.append(outputs)

torch.save({"classes": torch.stack(val_classes),
			"backgrounds": torch.stack(val_backgrounds),
			"activations": torch.stack(val_activations)}, "d3s_val_activations.pth")

