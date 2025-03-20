# ------------------------------------------------------------------------------
# Copyright (c) TokenSet authors
#
# Demo of set tokenizer
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import fire
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from train import center_crop_arr
from set_tokenizer.model import SetTokenizer

font = ImageFont.load_default(size=20)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.image_filenames = [
            f for f in os.listdir(root_dir) 
            if os.path.isfile(os.path.join(root_dir, f))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def reconstruction(tokenizer_path, input_folder, output_folder):
    device = 'cuda'
    os.makedirs(output_folder, exist_ok=True)

    tokenizer = SetTokenizer(
        token_size=12, \
        num_latent_tokens=128, \
        codebook_size=4096).cuda().eval()

    tokenizer.to(device)
    tokenizer.eval()

    tokenizer.load_state_dict(torch.load(tokenizer_path, map_location="cpu"), strict=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolderDataset(root_dir=input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for x, img_name in dataloader:
        x = x.to(device)

        with torch.no_grad():
            _, set_tokens = tokenizer.encode(x)
        set_tokens = set_tokens.reshape(x.shape[0], 1, -1)

        bs = set_tokens.size(0)
        seq_len = set_tokens.size(2)

        tokens_reverse = set_tokens.flip(dims=[2])
        _, tokens_shuffle_idx = torch.rand(bs, seq_len).sort(dim=1)
        tokens_shuffle = torch.gather(set_tokens, 2, tokens_shuffle_idx.unsqueeze(1).to(device))
        tokens_sorted_asc, _ = torch.sort(set_tokens, dim=2, descending=False)
        tokens_sorted_desc, _ = torch.sort(set_tokens, dim=2, descending=True)

        set_tokens_permutes = torch.cat((
            set_tokens,
            tokens_reverse,
            tokens_shuffle,
            tokens_sorted_asc,
            tokens_sorted_desc
        ), dim=1)
        set_tokens_permutes = set_tokens_permutes.reshape(-1, 1, seq_len)

        samples = tokenizer.decode_code(set_tokens_permutes)
    
        ori_image = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)

        labels = [
            "Original image",
            "Original order",
            "Reversed order",
            "Random shuffle",
            "Sorted ascending",
            "Sorted descending"
        ]

        images = []
        images.append(np.array(ori_image[0]))
        for i in range(5):
            images.append(np.array(samples[i]))

        concatenated_image_np = np.hstack(images)
        concatenated_image = Image.fromarray(concatenated_image_np)

        draw = ImageDraw.Draw(concatenated_image)
        single_img_width = images[0].shape[1]

        for idx, label in enumerate(labels):
            x = idx * single_img_width + 10
            y = 10
            draw.text((x, y), label, fill=(255, 0, 0), font=font)

        concatenated_image.save(f"{output_folder}/output_{img_name[0]}")


fire.Fire(reconstruction)