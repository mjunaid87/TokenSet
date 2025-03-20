# --------------------------------------------------------
# Sampling Module
# 
# Based on DiT (https://github.com/facebookresearch/DiT)
# Fixed-sum discrete diffusion is written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import argparse
import os
import re
import time
import torch_fidelity
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist

from flows import FSDD
from models import DiT_models
from train import quantize_tensor, force_fp32
from set_tokenizer.model import SetTokenizer


def find_model(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    checkpoint = checkpoint["ema"] # checkpoint["model"]
    return checkpoint


def counts_to_x2_with_fill(counts, total=64):
    """
    将 counts 转换回 x，并在需要时填充额外的类别索引。

    参数:
    counts (torch.Tensor): 形状为 [batch_size, num_classes, 1] 的张量，表示每个类别的计数。
    total (int): 每个样本中 x 的长度，默认为 64。

    返回:
    torch.Tensor: 形状为 [batch_size, total] 的张量，每个元素为对应的类别索引。
    """
    remains = counts.squeeze(-1).clone()
    counts[counts < 1] = 0

    # 预处理 counts
    counts = counts.squeeze(-1).round().long()  # [batch_size, num_classes]
    batch_size, num_classes = counts.shape
    device = counts.device

    # 计算每个样本的总计数
    sum_counts = counts.sum(dim=1)  # [batch_size]
    # print(sum_counts)
    remaining = (total - sum_counts).clamp(min=0)  # [batch_size]
    remains[remains >= 1] = 0

    sorted_order = torch.argsort(remains, dim=1, descending=True)  # [5, 4096]
    rank_indices = torch.argsort(sorted_order, dim=1)  # [5, 4096]
    mask = rank_indices < remaining.unsqueeze(1)  # [5, 4096]
    remain_result = mask.to(remains.device).to(torch.int32)
    counts += remain_result

    cum_counts = counts.cumsum(dim=1)  # [batch_size, num_classes]
    positions = torch.arange(total, device=device).unsqueeze(0).expand(batch_size, total)
    x = torch.searchsorted(cum_counts.contiguous(), positions.contiguous(), right=True)  # [batch_size, total]
   
    return x, sum_counts


def main(args):
    """
    Run sampling.
    """
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = DiT_models[args.model](
        in_channels=1, 
        token_len=args.codebook_size,
        num_classes=args.num_classes, 
        output_channels=args.token_len + 1
    ).to(device)

    if args.force_fp32:
        force_fp32(model, pattern=r".*embedder.*|.*final_layer.*|.*adaLN_modulation.*")
        
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    
    fsdd = FSDD("", model)

    tokenizer = SetTokenizer(
        token_size=args.tokenizer_embed_dim, \
        num_latent_tokens=args.token_len, \
        codebook_size=args.codebook_size).cuda().eval()

    checkpoint = torch.load(args.tokenizer_path, map_location="cpu")
    tokenizer.load_state_dict(checkpoint)

    assert args.cfg_scale >= 1.0 or args.cfg_threshold != '', "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0 or args.cfg_threshold != ''
    cfg_util = args.cfg_threshold if args.cfg_threshold != '' else args.cfg_scale

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pth", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-steps{args.num_sampling_steps}-" \
                  f"cfg{cfg_util}-image{args.num_fid_samples}-sigma{args.sigma_schedule}-" \
                  f"x0{args.sample_x0}-xt{args.sample_xt}-topk{args.top_k}-topp{args.top_p}-adjust{args.adjust_step}-name{args.name}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    batch_size = args.num_fid_samples // args.num_classes * args.class_batch
    assert args.num_fid_samples % args.num_classes == 0, "total_samples must be divisible by num_class"
    assert args.num_classes % (world_size * args.class_batch) == 0, "num_class must be divisible by world_size"
    iterations = int(args.num_classes // world_size // args.class_batch)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16}[args.mixed_precision]

    for j in pbar:
        
        fsdd.device = device
        z = torch.rand(batch_size, args.codebook_size).to(device)
        z = quantize_tensor(z, args.token_len).unsqueeze(1)
        z = z.permute(0, 2, 1)
        
        labels_gen = []
        for cls_idx in range(args.class_batch):
            labels_gen += [local_rank * iterations + args.class_batch * j + cls_idx] * (batch_size // args.class_batch)
        labels_gen = torch.tensor(labels_gen)
        y = torch.Tensor(labels_gen).long().cuda()

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, cfg_threshold=args.cfg_threshold)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward_softmax

        with torch.cuda.amp.autocast(enabled=(args.mixed_precision in ['bf16']), dtype=ptdtype):  
            traj_uncond  = fsdd.sample(
                sample_fn, z1=z, N=args.num_sampling_steps, \
                fixed_sum=args.token_len, \
                sigma_schedule=args.sigma_schedule, \
                sample_x0=args.sample_x0, sample_xt=args.sample_xt, \
                adjust_step=args.adjust_step, top_k=args.top_k, top_p=args.top_p,
                model_kwargs=model_kwargs, \
            )
        samples = traj_uncond[-1].float()
        
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples, sum_counts = counts_to_x2_with_fill(samples, total=args.token_len)
        samples = tokenizer.decode_code(samples)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = (local_rank * iterations + j) * batch_size + i
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

    dist.barrier()
    time.sleep(10)

    if rank == 0:
        input2 = None
        fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder_dir,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT1D-S")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=30)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--tokenizer_embed_dim', default=12, type=int, help='tokenizer output embedding dimension')
    parser.add_argument('--token_len', default=128, type=int, help='length of tokens')
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--class-batch", type=int, default=1)
    parser.add_argument("--cfg_threshold", type=str, default="4_50_55")
    parser.add_argument("--sigma_schedule", type=str, default="sigmoid100_0.8")
    parser.add_argument('--adjust_step', default=0.65, type=float)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.8, type=float)
    parser.add_argument("--sample_x0", type=str, default="topp")
    parser.add_argument("--sample_xt", type=str, default="multimonial")
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "bf16"]) 
    parser.add_argument('--force-fp32', action='store_true')
    args = parser.parse_args()
    main(args)
