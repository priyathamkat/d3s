import argparse
import glob
import os
import pickle

import numpy as np
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from main import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import trange
from utils import crop_to_squarize

_IMAGENET_ROOT = "/fs/cml-datasets/ImageNet/ILSVRC2012/"
_MASK_ROOT = "/cmlscratch/mmoayeri/data/hardImageNet/"

with open(
    "/cmlscratch/mmoayeri/hard_imagenet/data_collection/meta/idx_to_wnid.pkl", "rb"
) as f:
    idx_to_wnid = pickle.load(f)
wnid_to_idx = dict({v: k for k, v in idx_to_wnid.items()})

with open(
    "/cmlscratch/mmoayeri/hard_imagenet/data_collection/meta/hard_imagenet_idx.pkl",
    "rb",
) as f:
    inet_idx = pickle.load(f)


class HardImageNet(Dataset):
    def __init__(self, split="val", aug=None, ft=False, balanced_subset=False):
        """
        Returns original ImageNet index when ft is False, otherwise returns label between 0 and 14
        """
        self.aug = aug
        self.split = split
        self.balanced_subset = balanced_subset
        self.collect_mask_paths()
        # self.mask_paths = glob.glob(_MASK_ROOT + split+'/*')
        self.num_classes = 15
        self.ft = ft

    def map_wnid_to_label(self, wnid):
        ind = wnid_to_idx[wnid]
        if self.ft:
            ind = inet_idx.index(ind)
        return ind

    def collect_mask_paths(self):
        if self.balanced_subset and self.split == "train":
            # hard coded for now
            self.subset_size = 100

            with open(_MASK_ROOT + "paths_by_rank2.pkl", "rb") as f:
                ranked_paths = pickle.load(f)
            paths = []
            for c in ranked_paths:
                cls_paths = ranked_paths[c]
                paths += (
                    cls_paths[: self.subset_size] + cls_paths[(-1 * self.subset_size) :]
                )
            self.mask_paths = [
                _MASK_ROOT + "train/" + "_".join(p.split("/")[-2:]) for p in paths
            ]
            for p in self.mask_paths:
                if not os.path.exists(p):
                    self.mask_paths.remove(p)
        else:
            self.mask_paths = glob.glob(_MASK_ROOT + self.split + "/*")

    def __getitem__(self, ind):
        mask_path = self.mask_paths[ind]
        mask_path_suffix = mask_path.split("/")[-1]
        wnid = mask_path_suffix.split("_")[0]
        fname = mask_path_suffix[
            len(wnid) + 1 :
        ]  # if self.split == 'val' else mask_path_suffix

        img_path = os.path.join(_IMAGENET_ROOT, self.split, wnid, fname)
        img, mask = [Image.open(p) for p in [img_path, mask_path]]

        if self.aug:
            img = self.aug(img)
            mask = self.aug(mask)

        if img.shape[0] > 3:  # weird bug
            img, mask = [x[:3] for x in [img, mask]]

        class_ind = self.map_wnid_to_label(wnid)
        mask[mask > 0] = 1
        return img, mask, class_ind, mask_path_suffix

    def __len__(self):
        return len(self.mask_paths)


def make_batch(image, mask, device):

    mask = mask.unsqueeze(0)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = (1 - mask) * image
    if masked_image.shape[1] != 3:
        masked_image = masked_image.repeat([1, 3, 1, 1])

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs/inpainting",
        help="folder to save outputs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("../models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("../models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    aug = T.Compose([
        T.Lambda(lambda x: crop_to_squarize(x)),
        T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    hard_imagenet = HardImageNet(aug=aug)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            # for image, mask in tqdm(zip(images, masks)):
            for i in trange(len(hard_imagenet)):
                image, mask, _, mask_path_suffix = hard_imagenet[i]
                outpath = os.path.join(opt.outdir, f"{os.path.splitext(mask_path_suffix)[0]}.jpeg")
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1] - 1,) + c.shape[2:]
                samples_ddim, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    verbose=False,
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                inpainted = (1 - mask) * image + mask * predicted_image
                
                masked_image = (batch["masked_image"] + 1) / 2
                
                output = make_grid(torch.cat([masked_image, inpainted]))
                output = T.ToPILImage()(output)
                output.save(outpath)
