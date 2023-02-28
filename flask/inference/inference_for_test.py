import ray

from asyncio import PriorityQueue


# minDALLE
import sys
sys.path.append('minDALL-E')
sys.path.append('CLIP')

import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score

from skimage.util import view_as_blocks

# glide
from PIL import Image
from IPython.display import display
import torch as th

import sys
sys.path.append('glide-text2im')

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
   create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# stable-diffusion
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

import sys
sys.path.append('stable-diffusion')
sys.path.append('stable-diffusion/src/clip')
sys.path.append('stable-diffusion/src/taming-transformers')
sys.path.append('taming-transformers')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from PIL import Image

from transformers import logging
logging.set_verbosity_error()


# stable-diffusion 2
import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
#from torch.cuda.amp import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
#KBN
import sys
sys.path.append('../flask/stablediffusion2/ldm')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


# karlo
from typing import Iterator
import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode

import os
import logging
import sys
sys.path.append('/karlo/karlo')
sys.path.append('/karlo/karlo/sampler')
from template import BaseSampler, CKPT_PATH, SAMPLING_CONF
from models.clip import CustomizedCLIP, CustomizedTokenizer
from models.prior_model import PriorDiffusionModel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from queue import Queue
from torch._tensor import Tensor
####################################################################################################
class minDALLE(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(minDALLE, cls, *args, **kwargs).__new__(cls, *args, **kwargs)

        return cls.instance
    
    """
    #3장일 때 
    def refactor1(self, im_in, ncols=3):
        print()
        n,h,w,c = im_in.shape
        dn = (-n)%ncols # trailing images
        im_out = (np.empty((n+dn)*h*w*c,im_in.dtype)
           .reshape(-1,w*ncols,c))
        view=view_as_blocks(im_out,(h,w,c))
        for k,im in enumerate( list(im_in) + dn*[0] ):
            view[k//ncols,k%ncols,0] = im
        return im_out
    """
    #1장일 때 
    def refactor2(self, im_in, ncols=1):
        print()
        n,h,w,c = im_in.shape
        dn = (-n)%ncols # trailing images
        im_out = (np.empty((n+dn)*h*w*c,im_in.dtype)
           .reshape(-1,w*ncols,c))
        view=view_as_blocks(im_out,(h,w,c))
        for k,im in enumerate( list(im_in) + dn*[0] ):
            view[k//ncols,k%ncols,0] = im
        return im_out
    
    def load(self):
        device = 'cuda:0'
        set_seed(0)
        self.model = Dalle.from_pretrained('minDALL-E/1.3B') 
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=device)
       

    def txt2img(self, prompt):
        device = 'cuda:0'
        set_seed(0)

        #model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
        self.model.to(device=device)

        # Sampling
        images = self.model.sampling(prompt=prompt,
                        top_k=3, #256, # It is recommended that top_k is set lower than 256.
                        top_p=None,
                        softmax_temperature=1.0,
                        num_candidates=10, #96,
                        device=device).cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))

        # CLIP Re-ranking
        #model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        self.model_clip.to(device=device)
        rank = clip_score(prompt=prompt,
                  images=images,
                  model_clip=self.model_clip,
                 preprocess_clip=self.preprocess_clip,
                 device=device)

        images = images[rank]
        images = images[:1,:,:,:] #3장일때 1->3
        print(images.shape)

        images = self.refactor2(images) #3장일때 refactor1

        # to image
        outpath = 'outputs_test/minDALLE/' 
        grid_count = len(os.listdir(outpath)) - 1

        img = Image.fromarray((images*255).astype(np.uint8))
        save_file = os.path.join(outpath, f'grid-{grid_count:04}.png')
        img.save(save_file)
        grid_count += 1

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
            

        return save_file

####################################################################################################
class GLIDE(object):

    device = None
    options = None
    model = None
    diffusion = None
    model_up = None
    diffusion_up = None
    guidance_scale = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GLIDE, cls, *args, **kwargs).__new__(cls, *args, **kwargs)

        return cls.instance

    def load(self):
        has_cuda = th.cuda.is_available()
        #device = th.device('cpu' if not has_cuda else 'cuda')
        self.device = 'cuda:0'

        # Create base model.
        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = has_cuda
        self.options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.model.eval()
        if has_cuda:
            self.model.convert_to_fp16()
        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint('base', self.device))
        #print('total base parameters', sum(x.numel() for x in model.parameters()))

        self.guidance_scale = 3.0

        # Create upsampler model.
        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up['use_fp16'] = has_cuda
        self.options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)
        self.model_up.eval()
        if has_cuda:
            self.model_up.convert_to_fp16()
        self.model_up.to(self.device)
        self.model_up.load_state_dict(load_checkpoint('upsample', self.device))
        #print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))


    def show_images(self, batch: th.Tensor):
        """ Display a batch of images inline. """
        scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        #display(Image.fromarray(reshaped.numpy()))
        return reshaped

    # Create a classifier-free guidance sampling function
    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)


    def txt2img(self, prompt):
        # Sampling parameters
        batch_size = 1 #3장일때 3, 1장일때 1

        # Tune this parameter to control the sharpness of 256x256 images.
        # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
        upsample_temp = 0.997


        ##############################
        # Sample from the base model #
        ##############################

        # Create the text tokens to feed to the model.
        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=self.device,
            ),
        )
        # Sample from the base model.
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            self.model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.model.del_cache()

        # Show the output
        #show_images(samples)



        ##############################
        # Upsample the 64x64 samples #
        ##############################

        tokens = self.model_up.tokenizer.encode(prompt)
        tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
            tokens, self.options_up['text_ctx']
        )
        
        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=self.device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.model_up.del_cache()
        up_shape = (batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.diffusion_up.ddim_sample_loop(
            self.model_up,
            up_shape,
            noise=th.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.model_up.del_cache()

        # Show the output
        #show_images(up_samples)
                        
        # additionally, save as grid
        up_samples = self.show_images(up_samples)

        #grid = torch.stack(up_samples,0)
        #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        #grid = make_grid(grid, nrow=n_rows)

        # to image
        outpath = 'outputs_test/GLIDE/' 
        grid_count = len(os.listdir(outpath)) - 1
        #grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = up_samples.cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        save_file = os.path.join(outpath, f'grid-{grid_count:04}.png')
        img.save(save_file)
        grid_count += 1

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

        return save_file
# End of class

####################################################################################################
class StableDiffusion(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StableDiffusion, cls, *args, **kwargs).__new__(cls, *args, **kwargs)

        return cls.instance

    # load safety model
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


    def chunk(self,it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())


    def numpy_to_pil(self,images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def load_model_from_config(self, config, ckpt, verbose=False):
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

        device = 'cuda:0'
        model.cuda()
        model = model.to(device)
        model.eval()
        return model


    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img


    def load_replacement(self,x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x


    def check_safety(self,x_image):
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def parse(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render"
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs_test/sd1"
        )
        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--laion400m",
            action='store_true',
            help="uses the LAION400M model",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=512,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=512,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )

        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="how many samples to produce for each given prompt. A.k.a. batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=7.5,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            #default="/mnt/data/stable-diffusion/model.ckpt",
            default="/home/user/Github/streamlit/sd-v1-4.ckpt",
            help="path to checkpoint of model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )

        opt = parser.parse_args()
        return opt

    def load(self, opt):

        seed_everything(opt.seed)


        config = OmegaConf.load(f"{opt.config}")
        model = self.load_model_from_config(config, f"{opt.ckpt}")

        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = 'cuda:0'
        model = model.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext


        return model, sampler, outpath, wm_encoder, sample_path, base_count, grid_count, start_code, precision_scope



    def txt2img(self, input_str, opt, model, sampler, outpath, wm_encoder, sample_path, base_count, grid_count, start_code, precision_scope):


        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = input_str #opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))


        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = self.put_watermark(img, wm_encoder)
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)
                        
                        grid_count = len(os.listdir(outpath)) - 1

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = self.put_watermark(img, wm_encoder)
                        save_file = os.path.join(outpath, f'grid-{grid_count:04}.png')
                        img.save(save_file)
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

        return save_file


# End of class

####################################################################################################
class StableDiffusion2(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StableDiffusion2, cls, *args, **kwargs).__new__(cls, *args, **kwargs)

        return cls.instance


    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
    
    
    def load_model_from_config(self, config, ckpt, verbose=False):
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


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a professional photograph of an astronaut riding a triceratops",
            help="the prompt to render"
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs_test/sd2"
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--dpm",
            action='store_true',
            help="use DPM (2) sampler",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across all samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=512,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=512,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor, most often 8 or 16",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="how many samples to produce for each given prompt. A.k.a batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=9.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file, separated by newlines",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="stablediffusion2/configs/stable-diffusion/v2-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="/home/user/Github/streamlit/stablediffusion2/v2-1_512-ema-pruned.ckpt",
            help="path to checkpoint of model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )
        parser.add_argument(
            "--repeat",
            type=int,
            default=1,
            help="repeat each prompt in file this often",
        )
        opt = parser.parse_args()
        return opt


    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img

    def load(self, opt):
        seed_everything(opt.seed)

        config = OmegaConf.load(f"{opt.config}")
        model = self.load_model_from_config(config, f"{opt.ckpt}")

        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = 'cuda:0'
        model = model.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        elif opt.dpm:
            sampler = DPMSolverSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "SDV2"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
        

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext

        return model, sampler, outpath, wm_encoder, sample_path, sample_count, base_count, grid_count, start_code, precision_scope


    #sample_count 추가
    def txt2img(self, input_str, opt, model, sampler, outpath, wm_encoder, sample_path, sample_count, base_count, grid_count, start_code, precision_scope):

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = input_str #opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = [p for p in data for i in range(opt.repeat)]
                data = list(chunk(data, batch_size))

        #yjlee
        #with torch.no_grad(), \
        #    precision_scope("cuda"), \
        #    model.ema_scope():
        with torch.no_grad(), \
            precision_scope("cuda"), \
            model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples, _ = sampler.sample(S=opt.steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = self.put_watermark(img, wm_encoder)
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1
                            sample_count += 1

                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                grid_count = len(os.listdir(outpath)) - 1
                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid = self.put_watermark(grid, wm_encoder)
                save_file = os.path.join(outpath, f'grid-{grid_count:04}.png')
                grid.save(save_file)
                grid_count += 1


        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.") 


        return save_file

# End of class

####################################################################################################
class Karlo(BaseSampler):
    def __init__(
        self,
        root_dir = "../karlo",
        sampling_type = "default",
    ):
        super().__init__(root_dir, sampling_type)

        clip_path : str = "../karlo/ViT-L-14.pt"
        clip = CustomizedCLIP.load_from_checkpoint(clip_path)
    
        clip = torch.jit.script(clip)
        clip.cuda()
        clip.eval()
        self._clip = clip
        self._tokenizer = CustomizedTokenizer()
        ###############################################################
        ckpt_path : str = "prior-ckpt-step=01000000-of-01000000.ckpt",
        #clip_stat_path = "ViT-L-14_stats.th",
        logging.info(f"Loading prior: {ckpt_path}")

        #config = OmegaConf.load("configs/prior_1B_vit_l.yaml")
        config = OmegaConf.load("../flask/karlo/configs/prior_1B_vit_l.yaml")
        clip_mean, clip_std = torch.load(
            "../flask/karlo/ViT-L-14_stats.th", map_location="cpu"
        )
    
        prior = PriorDiffusionModel.load_from_checkpoint(
            config,
            CustomizedTokenizer(),
            clip_mean,
            clip_std,
            #os.path.join(self._root_dir, ckpt_path),
            "../flask/karlo/prior-ckpt-step=01000000-of-01000000.ckpt",
            strict=True,
        )
        prior.cuda()
        prior.eval()
        logging.info("done.")
        self._prior = prior 
        ################################################################
        ckpt_path2 = "decoder-ckpt-step=01000000-of-01000000.ckpt",
        logging.info(f"Loading decoder: {ckpt_path2}")
        config = OmegaConf.load("../flaskb/karlo/configs/decoder_900M_vit_l.yaml")
        decoder = self._DECODER_CLASS.load_from_checkpoint(
            config,
            self._tokenizer,
            "../flask/karlo/decoder-ckpt-step=01000000-of-01000000.ckpt",
            strict=True,
        )
        decoder.cuda()
        decoder.eval()
        logging.info("done.")

        self._decoder = decoder
        ################################################################
        ckpt_path3 = "improved-sr-ckpt-step=1.2M.ckpt",
        logging.info(f"Loading SR(64->256): {ckpt_path3}")

        config = OmegaConf.load("../flask/karlo/configs/improved_sr_64_256_1.4B.yaml")
        sr = self._SR256_CLASS.load_from_checkpoint(
            config, "../flask/karlo/improved-sr-ckpt-step=1.2M.ckpt", strict=True
        )
        sr.cuda()
        sr.eval()
        logging.info("done.")

        self._sr_64_256 = sr




    @classmethod
    def from_pretrained(
        cls,
        root_dir : str,
        clip_model_path = "../flask/karlo/ViT-L-14.pt",
        clip_stat_path = "../flask/karlo/ViT-L-14_stats.th",
        sampling_type = "default",
    ):

        model = cls(
            root_dir=root_dir,
            sampling_type=sampling_type,
        )
        model.load_clip(clip_model_path)
        model.load_prior(
            f"{CKPT_PATH['prior']}",
            clip_stat_path=clip_stat_path,
        )
        model.load_decoder(f"{CKPT_PATH['decoder']}")
        model.load_sr_64_256(CKPT_PATH["sr_256"])

        return model   

    def preprocess(
        self,
        prompt: str,
        bsz: int,

    ):
        """Setup prompts & cfg scales"""
        prompts_batch = [prompt for _ in range(bsz)]

        prior_cf_scales_batch = [self._prior_cf_scale] * len(prompts_batch)
        prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda")

        decoder_cf_scales_batch = [self._decoder_cf_scale] * len(prompts_batch)
        decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda")
        
        """Get CLIP text feature"""
        clip_model = self._clip
        tokenizer = self._tokenizer
        max_txt_length = self._prior.model.text_ctx

        tok, mask = tokenizer.padded_tokens_and_mask(prompts_batch, max_txt_length)
        cf_token, cf_mask = tokenizer.padded_tokens_and_mask([""], max_txt_length)
        if not (cf_token.shape == tok.shape):
            cf_token = cf_token.expand(tok.shape[0], -1)
            cf_mask = cf_mask.expand(tok.shape[0], -1)

        tok = torch.cat([tok, cf_token], dim=0)
        mask = torch.cat([mask, cf_mask], dim=0)

        tok, mask = tok.to(device="cuda"), mask.to(device="cuda")
        txt_feat, txt_feat_seq = clip_model.encode_text(tok)

        return (
            prompts_batch,
            prior_cf_scales_batch,
            decoder_cf_scales_batch,
            txt_feat,
            txt_feat_seq,
            tok,
            mask,
        )

    def __call__(
        self,
        prompt: str,
        bsz: int,
        progressive_mode="final",
) -> Iterator[torch.Tensor]:
        assert progressive_mode in ("loop", "stage", "final")
        with torch.no_grad(), torch.cuda.amp.autocast():
            (
                prompts_batch,
                prior_cf_scales_batch,
                decoder_cf_scales_batch,
                txt_feat,
                txt_feat_seq,
                tok,
                mask,
            ) = self.preprocess(
                prompt,
                bsz,
            )

            """ Transform CLIP text feature into image feature """
            img_feat = self._prior(
                txt_feat,
                txt_feat_seq,
                mask,
                prior_cf_scales_batch,
                timestep_respacing=self._prior_sm,
            )

            """ Generate 64x64px images """
            images_64_outputs = self._decoder(
                txt_feat,
                txt_feat_seq,
                tok,
                mask,
                img_feat,
                cf_guidance_scales=decoder_cf_scales_batch,
                timestep_respacing=self._decoder_sm,
            )
            
            images_64 = None
            for k, out in enumerate(images_64_outputs):
                images_64 = out
            if progressive_mode == "loop":
                    yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)
            if progressive_mode == "stage":
                yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)

            images_64 = torch.clamp(images_64, -1, 1)

            """ Upsample 64x64 to 256x256 """
            images_256 = TVF.resize(
                images_64,
                [256, 256],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            images_256_outputs = self._sr_64_256(
                images_256, timestep_respacing=self._sr_sm
            )
            
            for k, out in enumerate(images_256_outputs):
                images_256 = out
                if progressive_mode == "loop":
                    yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)
            if progressive_mode == "stage":
                yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)   
        yield torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)



    def tensor_to_images(self, tensor: torch.Tensor, output_res=(1024, 1024)):
        assert tensor.ndim == 4
        tensor = torch.clone(tensor)
        # NCHW -> NHWC
        images = torch.permute(tensor * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
        concat_image = np.concatenate(images, axis=1)
        target_size = (output_res[1] * tensor.shape[0], output_res[0])
        print()
        concat_image = Image.fromarray(concat_image).resize(
            target_size, resample=Image.NEAREST
        )
        return images, concat_image
     


   
    def _sample(self, output_generator):
        iterator = iter(output_generator)
        out = iterator.__next__()
        images, concat_image = self.tensor_to_images(out, (256, 256))
        outpath = 'outputs_test/karlo/' 
        #concat_image.save('karlo_image','png')
        grid_count = len(os.listdir(outpath)) - 1
        save_file = os.path.join(outpath, f'grid-{grid_count:04}.png')
        concat_image.save(save_file)
        grid_count += 1

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

        return save_file


