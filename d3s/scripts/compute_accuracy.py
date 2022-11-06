import json
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import timm
import torch
from absl import app, flags
from d3s.datasets import D3S, ImageNet
from matplotlib.font_manager import FontProperties, fontManager
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_enum(
    "shift",
    "background-shift",
    ["all", "background-shift", "geography-shift", "time-shift"],
    "Shift to use for evaluation",
)
flags.DEFINE_integer("batch_size", 128, "Batch size for evaluating the model")
flags.DEFINE_bool(
    "save_majority_misclassified",
    False,
    "Save images misclassified by majority of models",
)


@torch.no_grad()
def test(model, dataloader, desc):
    model.eval()
    top1 = top5 = total = 0
    misclassified = []
    for batch in tqdm(dataloader, leave=False, desc=desc, file=sys.stdout):
        images, labels = batch[:2]
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, top5_indices = outputs.topk(5, dim=1)
        top1_indices = top5_indices[:, 0]
        top5 += (top5_indices == labels.unsqueeze(1)).sum().item()
        is_correct = top1_indices == labels
        incorrect_indices = total + torch.nonzero(~is_correct, as_tuple=True)[0]
        misclassified.extend(incorrect_indices.tolist())
        top1 += is_correct.sum().item()
        total += labels.shape[0]
    return 100 * top1 / total, 100 * top5 / total, misclassified


def main(argv):
    data = defaultdict(list)
    num_incorrect_classifications = defaultdict(int)
    model_names = ["resnet50", "resnext50_32x4d", "densenet121", "efficientnet_b1", "vit_base_patch16_224", "inception_resnet_v2", "mobilenetv3_small_100"]
    for model_name in model_names:
        data["Model Architecture"].append(model_name)
        model = timm.create_model(model_name, pretrained=True).cuda()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        dataset = ImageNet(split="val", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        desc = f"Testing {model_name} on ImageNet"
        imagenet_top1, imagenet_top5, _ = test(model, dataloader, desc=desc)

        data["ImageNet Top-1"].append(imagenet_top1)
        data["ImageNet Top-5"].append(imagenet_top5)
        
        dataset = D3S(Path(FLAGS.d3s_root), split="val", shift=FLAGS.shift, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        desc = f"Testing {model_name} on D3S"
        d3s_top1, d3s_top5, misclassified = test(model, dataloader, desc=desc)
        if FLAGS.save_majority_misclassified:
            for idx in misclassified:
                num_incorrect_classifications[idx] += 1

        data["D3S Top-1"].append(d3s_top1)
        data["D3S Top-5"].append(d3s_top5)

    df = pd.DataFrame(data=data)
    df["Drop in Top-1"] = df["ImageNet Top-1"] - df["D3S Top-1"]
    df["Drop in Top-5"] = df["ImageNet Top-5"] - df["D3S Top-5"]
    save_folder = Path(__file__).parent.parent / "outputs"
    df.to_csv(save_folder / "accuracy.csv", index=False)
    
    melted = pd.melt(df, id_vars=["Model Architecture"], value_vars=["Drop in Top-1", "Drop in Top-5"], var_name="type", value_name="Drop")
    
    path = "/cmlscratch/pkattaki/void/d3s/d3s/assets/Roboto-Regular.ttf"
    fontManager.addfont(path)
    prop = FontProperties(fname=path, weight="regular")
    sns.set(font=prop.get_name())
    plt.figure(dpi=600)
    plot = sns.catplot(data=melted, kind="bar", x="Model Architecture", y="Drop", hue="type", facet_kws={'legend_out': True})
    legend_labels = ["Top-1", "Top-5"]
    for legend_text, legend_label in zip(plot._legend.texts, legend_labels):
        legend_text.set_text(legend_label)
    plot._legend.set_title("")
    plt.ylabel("Accuracy Drop on D3S")
    plt.savefig(save_folder / "accuracy.png")
    
    if FLAGS.save_majority_misclassified:
        majority_misclassified = []
        for idx, count in num_incorrect_classifications.items():
            if count == len(model_names):
                majority_misclassified.append(
                    {
                        "image": dataset.images[idx]["image_path"],
                        "classIdx": dataset.images[idx]["class_idx"],
                    }
                )
        with open(
            Path(__file__).parents[2]
            / "quality-control/src/assets/input/inputData.json",
            "w",
        ) as f:
            json.dump(majority_misclassified, f)


if __name__ == "__main__":
    app.run(main)
