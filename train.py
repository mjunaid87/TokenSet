# --------------------------------------------------------
# Training Module
# 
# Based on DiT (https://github.com/facebookresearch/DiT)
# Modifications including fixed-sum discrete diffusion and BF16 support
# are written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
from torch import Tensor
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
import logging
import os
import datetime
import re
from time import time
from glob import glob
import numpy as np
from PIL import Image
from copy import deepcopy
from collections import defaultdict, OrderedDict
from types import MethodType

from flows import FSDD, quantize_tensor, greedy_multinomial_sample
from models import DiT_models
from set_tokenizer.model import SetTokenizer

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def cast_to_fp32(args):
    if isinstance(args, list):
        return [cast_to_fp32(arg) for arg in args]
    elif isinstance(args, tuple):
        return tuple(cast_to_fp32(arg) for arg in args)
    elif isinstance(args, dict):
        return {k: cast_to_fp32(v) for k, v in args.items()}
    elif isinstance(args, torch.Tensor):
        if args.dtype == torch.float16 or args.dtype == torch.bfloat16:
            return args.to(torch.float32)
        else:
            return args
    else:
        return args


def force_fp32_forward(module: nn.Module):
    """force model forward to use fp32."""
    old_forward = module.forward

    def new_forward(self, *args, **kwargs):
        # cast all inputs to fp32
        args = cast_to_fp32(args)
        kwargs = cast_to_fp32(kwargs)
        with torch.autocast(device_type="cuda", enabled=False):
            return old_forward(*args, **kwargs)

    module.forward = MethodType(new_forward, module)


def force_fp32(
    model: nn.Module, pattern: str, prefix: str = None, logger: logging.Logger = None
):
    for name, module in model.named_children():
        full_name = name if prefix is None else ".".join([prefix, name])
        if re.match(pattern, full_name):
            if logger is not None:
                logger.info(f"Forcing {full_name} to execute in fp32 mode.")
            # do something
            force_fp32_forward(module)
        else:
            force_fp32(module, pattern, full_name, logger)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600),)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(os.path.join(experiment_dir, "experiments_logger"))
    else:
        logger = create_logger(None)

    # Create model:
    model = DiT_models[args.model](
        in_channels=1, token_len=args.codebook_size,
        num_classes=args.num_classes,
        output_channels=args.token_len + 1
    )
        
    if args.force_fp32:
        force_fp32(model, pattern=r".*embedder.*|.*final_layer.*|.*adaLN_modulation.*", logger=logger)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])

    fsdd = FSDD("", model)
    
    tokenizer = SetTokenizer(
        token_size=args.tokenizer_embed_dim, 
        num_latent_tokens=args.token_len, 
        codebook_size=args.codebook_size).cuda().eval()
    checkpoint = torch.load(args.tokenizer_path, map_location="cpu")
    tokenizer.load_state_dict(checkpoint)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(os.path.join(args.data_path), transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if args.resume is not None:   
        ckpt = torch.load(args.resume, map_location="cpu")       
        # ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)        
        model.module.load_state_dict(ckpt["model"])  
        logger.info(f"Load resume checkpoint from {args.resume}")
        ema.load_state_dict(ckpt["ema"]) 
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt["steps"] if "steps" in ckpt else int(args.resume.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        logger.info(f"Resume training from checkpoint: {args.resume}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        del ckpt
    else:
        train_steps = 0
        start_epoch = 0
        
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    bins = 10
    name2counts = defaultdict(list)
    name2val = defaultdict(float)  # values this iteration
    name2cnt = defaultdict(int)
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                _, tokenize_codes = tokenizer.encode(x)
                x = tokenize_codes.squeeze()

                # Dual Transformation
                counts = torch.zeros((x.shape[0], args.codebook_size), device=x.device, dtype=torch.long)
                counts.scatter_add_(dim=1, index=x, src=torch.ones_like(x, dtype=torch.long))
                counts = counts.unsqueeze(1).float()

            loss_dict = {}
            counts = counts.to(torch.long)

            fsdd.device = x.device
            z = torch.rand(counts.shape[0], args.codebook_size).to(x.device)
            z_q = quantize_tensor(z, args.token_len).unsqueeze(1)

            counts, z_q = counts.permute(0, 2, 1), z_q.permute(0, 2, 1)

            z_t, t = fsdd.get_train_tuple(z0=z_q, z1=counts)
            
            delta = torch.abs(z_q - counts) 

            sigma = delta * 1.0 / 4
            epsilon = 1e-6
            sigma = sigma + epsilon

            k_values = torch.arange(args.token_len, dtype=z_q.dtype, device=z_q.device).view(1, 1, args.token_len)  
            exponent = -0.5 * ((k_values - z_t) / sigma) ** 2
            sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=z_q.dtype, device=z_q.device))
            pdf = 1.0 / (sigma * sqrt_2pi) * torch.exp(exponent)
            pdf_sum = pdf.sum(dim=-1, keepdim=True)
            pdf_normalized = pdf / pdf_sum 
            z_tq = greedy_multinomial_sample(pdf_normalized, args.token_len)
            with torch.cuda.amp.autocast(enabled=(args.mixed_precision in ['bf16']), dtype=ptdtype):  
                logits = model(z_tq, t.squeeze(), y.squeeze())
            logits = logits.permute(0, 2, 1)

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss_fm = loss_fn(logits, counts.squeeze())

            for t_, loss_ in zip(t.squeeze().cpu().numpy(), torch.mean(loss_fm, dim=(1)).detach().cpu().numpy()):
                quartile = int(bins * t_)
                oldval, cnt = name2val[f"q{quartile}"], name2cnt[f"q{quartile}"]
                name2val[f"q{quartile}"] = oldval * cnt / (cnt + 1) + loss_ / (cnt + 1)
                name2cnt[f"q{quartile}"] = cnt + 1

            loss_dict['loss'] = loss_fm
            
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                show_keys = ['q' + str(i) for i in range(bins)]
                bin_loss = ','.join([f"{name2val[key]*1000:0.4f}" for key in show_keys])
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Bin Loss: {bin_loss}, Train Steps/Sec: {steps_per_sec:.2f}")
                if dist.get_rank() == 0: 
                    writer.add_scalar('Loss/train', loss.item(), train_steps)
                    for name, bin_value in name2val.items():
                        writer.add_scalar('train_loss/'+name, bin_value, train_steps)  
                    
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                name2val = defaultdict(float)  # values this iteration
                name2cnt = defaultdict(int)
                break

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
        break
        
    model.eval()  
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT1D-S")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--tokenizer_embed_dim', default=12, type=int, help='tokenizer output embedding dimension')
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument('--token_len', default=128, type=int, help='length of tokens')
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "bf16"]) 
    parser.add_argument('--force-fp32', action='store_true')
    args = parser.parse_args()
    main(args)
