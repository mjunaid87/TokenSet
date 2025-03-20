# ------------------------------------------------------------------------------
# Copyright (c) TokenSet authors
#
# Fixed-sum discrete diffusion
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
from torch import Tensor
from tqdm import tqdm


def quantize_tensor(x, token_len):
    floor_x = torch.floor(x).long()  # shape: (bs, codebook_size)
    residual = token_len - floor_x.sum(dim=1, keepdim=True).squeeze(1)  # shape: (bs,)
    
    frac = x - floor_x.float()  # shape: (bs, codebook_size)
    
    sorted_frac, order = torch.sort(frac, dim=1, descending=True)
    ranks = torch.argsort(order, dim=1)
    
    allocation = (ranks < residual.unsqueeze(1)).long()  # shape: (bs, codebook_size)
    quant_x = floor_x + allocation  # shape: (bs, codebook_size)
    return quant_x


def greedy_adjust(z0, probs, fix_number):

    bs, len_seq, class_num = probs.size()  
    z0_sum = z0.sum(dim=1)  # shape: [bs]

    adjust_num = fix_number - z0_sum  # shape: [bs], 
    p0 = probs.gather(2, z0.unsqueeze(-1)).squeeze(-1)  # shape: [bs, len]

    valid_prev = (z0 > 0)  # mask
    z0_prev = (z0 - 1).clamp(min=0)
    p_prev = probs.gather(2, z0_prev.unsqueeze(-1)).squeeze(-1)
    delta_decrease = torch.zeros(z0.shape).to(z0.device)
    delta_decrease[valid_prev] = (p_prev - p0)[valid_prev]
    delta_decrease[~valid_prev] = -float('inf')

    valid_next = (z0 < class_num - 1)  # mask
    z0_next = (z0 + 1).clamp(max=class_num - 1)
    p_next = probs.gather(2, z0_next.unsqueeze(-1)).squeeze(-1)
    delta_increase = torch.zeros(z0.shape).to(z0.device)
    delta_increase[valid_next] = (p_next - p0)[valid_next]
    delta_increase[~valid_next] = -float('inf')

    delta_decrease_sorted, indices_decrease = torch.sort(delta_decrease, dim=1, descending=True)
    delta_increase_sorted, indices_increase = torch.sort(delta_increase, dim=1, descending=True)

    z0_adjusted = z0.clone()

    adjust_negative = adjust_num < 0
    if adjust_negative.any():
        adjust_negative_index = adjust_negative.nonzero(as_tuple=False).squeeze()
        if len(adjust_negative_index.squeeze().shape) == 0:
          adjust_negative_index = adjust_negative_index.unsqueeze(0)
        num_to_adjust_negative = (-adjust_num[adjust_negative]).long()
        delta_decrease_sorted_neg = delta_decrease_sorted[adjust_negative]
        indices_decrease_neg = indices_decrease[adjust_negative]

        max_adjust_neg = min(num_to_adjust_negative.max().item(), len_seq)
        indices_to_adjust_neg = indices_decrease_neg[:, :max_adjust_neg]

        mask = torch.arange(max_adjust_neg)[None, :].to(z0.device) < num_to_adjust_negative[:, None]
        batch_indices = adjust_negative_index.unsqueeze(1).expand(-1, max_adjust_neg)
        batch_indices = batch_indices[mask]
        len_indices = indices_to_adjust_neg[mask]

        z0_adjusted[batch_indices, len_indices] -= 1

    adjust_positive = adjust_num > 0
    if adjust_positive.any():
        adjust_positive_index = adjust_positive.nonzero(as_tuple=False).squeeze()
        if len(adjust_positive_index.squeeze().shape) == 0:
          adjust_positive_index = adjust_positive_index.unsqueeze(0)
        num_to_adjust_positive = adjust_num[adjust_positive].long()
        delta_increase_sorted_pos = delta_increase_sorted[adjust_positive]
        indices_increase_pos = indices_increase[adjust_positive]

        max_adjust_pos = min(num_to_adjust_positive.max().item(), len_seq)
        indices_to_adjust_pos = indices_increase_pos[:, :max_adjust_pos]

        mask = torch.arange(max_adjust_pos)[None, :].to(z0.device) < num_to_adjust_positive[:, None]
        batch_indices = adjust_positive_index.unsqueeze(1).expand(-1, max_adjust_pos)
        batch_indices = batch_indices[mask]
        len_indices = indices_to_adjust_pos[mask]

        z0_adjusted[batch_indices, len_indices] += 1

    z0_adjusted = z0_adjusted.clamp(min=0, max=class_num - 1)

    return z0_adjusted


def greedy_topp_or_topk_sample(probs, fix_number, step=0, adjust_step=0.65, top_k=0, top_p=0.8):
    bs, seq_len, num_classes = probs.shape
    probs_flat = torch.zeros_like(probs).to(probs.device)

    if top_k >= 1:
      # Topk sampling
      topk_probs, topk_indices = probs.topk(top_k, dim=-1)
      probs_flat.scatter_(-1, topk_indices, topk_probs)
      probs_flat /= probs_flat.sum(dim=-1, keepdim=True)
    else:
      # TopP sampling
      sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
      cumulative_probs = sorted_probs.cumsum(dim=-1)
      mask = cumulative_probs <= top_p
      mask[..., 1:] = mask[..., :-1].clone()
      mask[..., 0] = True
      sorted_probs = sorted_probs * mask
      sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
      probs_flat.scatter_(-1, sorted_indices, sorted_probs)

    sampled = torch.multinomial(probs_flat.view(-1, num_classes), num_samples=1)
    z0 = sampled.view(bs, seq_len)

    if step >= adjust_step:
      return greedy_adjust(z0, probs, fix_number)
    else:
      return z0


def greedy_multinomial_sample(probs, fix_number):
    z0 = torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )

    return greedy_adjust(z0, probs, fix_number)


def sample_prob(probs, fix_number, sample_type='topp', step=0, adjust_step=0.65, top_k=0, top_p=0.8):
    if sample_type == 'topp' or sample_type == 'topk':
      return greedy_topp_or_topk_sample(probs, fix_number, step, adjust_step, top_k, top_p)
    else:
      return greedy_multinomial_sample(probs, fix_number)


class FSDD():
  def __init__(self, device, model=None):
    self.model = model
    self.device = device
  
  def get_train_tuple(self, z0=None, z1=None, eps = 1e-5):
    t = torch.rand((z1.shape[0], 1), device=self.device)
    t = t * (1 - eps) + eps
    if len(z1.shape) == 2:
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 3:
      t = t.view(-1, 1, 1)
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      z_t =  t * z1 + (1.-t) * z0
  
    return z_t, t

  @torch.no_grad()
  def sample(self, model, z1=None, N=None, fixed_sum=64, \
        use_tqdm=False, sigma_schedule='sigmoid75_0.84', \
        sample_x0='topp', sample_xt='multimonial', \
        adjust_step=0.65, top_k=0, top_p=0.8, model_kwargs=None):

    tq = tqdm if use_tqdm else lambda x: x

    traj = [] # to store the trajectory
    z = z1.detach().clone()
    batchsize = z.shape[0]

    u = torch.linspace(0, 1.0, steps=N+1, device=self.device)
    t_values = u

    traj.append(z.detach().clone())
    for i in tq(range(N)):
      t = torch.ones((batchsize, 1), device=self.device) * t_values[i]
      t_next = torch.ones((batchsize, 1, 1), device=self.device) * t_values[i+1]

      vt = model(z.squeeze(), t.squeeze(), **model_kwargs)
      z0 = sample_prob(vt, fixed_sum, sample_x0, t[0], adjust_step, top_k, top_p).unsqueeze(-1)
      traj.append(z0.detach().clone())
    
      z1_post = (z - t.unsqueeze(-1) * z0) / (1-t.unsqueeze(-1))
      z_t =  t_next * z0 + (1.-t_next) * z1_post

      delta = torch.abs(z1_post - z0) 
      
      if 'sigmoid' in sigma_schedule:
        k, inter = sigma_schedule.replace('sigmoid', '').split('_')
        k, inter = float(k), float(inter)
        sigmat = torch.tensor(-k * ((i+1) * 1.0 / N - inter), device=z0.device)
        sigmat = torch.sigmoid(sigmat)
      else:
        sigmat = 1.0

      sigma = (delta * 1.0 / 4) * sigmat
      epsilon = 1e-6
      sigma = sigma + epsilon

      try:
        k_values = torch.arange(fixed_sum, dtype=z_t.dtype, device=z_t.device).view(1, 1, fixed_sum)  
        exponent = -0.5 * ((k_values - z_t) / sigma) ** 2
        sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=z_t.dtype, device=z_t.device))
        pdf = 1.0 / (sigma * sqrt_2pi) * torch.exp(exponent)
        pdf_sum = pdf.sum(dim=-1, keepdim=True)
        pdf_normalized = pdf / pdf_sum 
        z = sample_prob(pdf_normalized, fixed_sum, sample_xt, t[0], adjust_step, top_k, top_p).unsqueeze(-1)
      except:
        z = quantize_tensor(z_t.squeeze(), fixed_sum).unsqueeze(-1)

      traj.append(z.detach().clone())
    return traj