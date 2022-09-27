import numpy as np
import torch
from einops import rearrange, repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import trange

CONFIG_PATH = "/cmlscratch/pkattaki/void/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
CKPT_PATH = "/cmlscratch/pkattaki/void/stable-diffusion/models/ldm/stable-diffusion-v4/sd-v1-4.ckpt"

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

class DiffusionGenerator():
    def __init__(self, seed=42, config_path=CONFIG_PATH, ckpt_path=CKPT_PATH):
        seed_everything(seed)

        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt_path)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        self.sampler = DDIMSampler(self.model)

        self.batch_size = 1
        self.n_iter = 1

        self.precision_scope = autocast



    def conditional_generate(self, prompt, init_image, strength, ddim_steps=50, ddim_eta=0.0):   
        scale = 5.0
        n_rows = 2
        data = [self.batch_size * [prompt]]

        init_image = init_image.to(self.device)
        all_samples = [(init_image + 1.0) / 2.0]
        init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling", disable=True):
                        for prompts in data:
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*self.batch_size).to(self.device))
                            # decode it
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, x0=init_image)

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            all_samples.append(x_samples)

                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
        return grid

    def unconditional_generate(self, prompt, ddim_steps=50, ddim_eta=0.0):
        scale = 7.5
        C = 4
        f = 8
        H = W = 512
        n_samples = 1
        n_rows = 1
        data = [self.batch_size * [prompt]]
        
        start_code = torch.randn([n_samples, C, H // f, W // f], device=self.device)
        
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(self.n_iter, desc="Sampling", disable=True):
                        for prompts in data:
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_samples_ddim = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                            all_samples.append(x_samples_ddim)
                                
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        grid = Image.fromarray(grid.astype(np.uint8))
        return grid
