# 🎨 Tokenize Image as a Set (TokenSet)
Official PyTorch implementation for our paper **"Tokenize Image as a Set"**, a novel framework for set-based image tokenization and fixed-sum discrete generative modeling.

[📄 **Paper PDF**](Your_paper_pdf_url_here)


## 🌟 Highlights of Our Approach

This paper introduces a fundamentally new paradigm for image generation through set-based tokenization and fixed-sum discrete modeling strategies:

- **Set-based Tokenization**: Represent images as permutation-invariant token sets rather than sequential codes.
- **Dual Transformation**: Bijectively map unordered token sets into fixed-length integer sequences, ensuring invertibility and consistency.
- **Fixed-Sum Discrete Diffusion (FSDD)**: The first discrete diffusion framework that simultaneously models discrete values, maintains fixed sequence lengths, and preserves summation constraints.

<p align="center">
    <img src="https://github.com/Gengzigang/gif-storage/blob/main/comparative_results_slow.gif" width="600px"/>
    <br/>
    <em>Animated visualization of iterative token replacement.</em>
</p>

---

## 📂 What's Included

✅ Well-trained **Set Tokenizer** and corresponding inference scripts for image reconstruction.  

✅ Well-trained **class-conditional generation models** (DiT-small and DiT-base based architectures).  

✅ Implementation of **Fixed-Sum Discrete Diffusion**, including both training and sampling code. 

---

## 🚀 Quick Start Guide

### Step 1: Clone the repository
```bash
git clone https://github.com/Gengzigang/TokenSet.git
cd TokenSet
```

### Step 2: Set up the environment
```bash
conda env create -f environment.yml
conda activate tokenset
```

### Step 3: Download pretrained models
Obtain the pretrained tokenizer and generation checkpoints from [Google Drive](https://drive.google.com/drive/folders/1wf05nt7TGDoQV6lj10-hxfZoW3HSk9rZ?usp=drive_link):
```
pretrained_models/
├── set_tokenizer_128_4096.pth
├── fsdd_small_128_4096.pth
└── fsdd_base_128_4096.pth
```

### Step 4: Run inference
- **Image reconstruction from tokenizer sets**:
```bash
python demo_tokenizer.py pretrained_models/set_tokenizer_128_4096.pth demo/ demo_output/
```

- **Class-conditional Generation** (DiT small/base):
Before evaluating with gFid, please first download the "fid_stats" folder from the [MAR](https://github.com/LTH14/mar/tree/main) repository, which contains reference statistics needed for evaluation.

```bash
# Small model – optimal for gFID
python -m torch.distributed.run --master_port 17828 --nproc_per_node=8 sample.py \
    --model DiT1D-S \
    --ckpt pretrained_models/fsdd_small_128_4096.pth \
    --tokenizer_path pretrained_models/set_tokenizer_128_4096.pth \
    --mixed-precision bf16 --force-fp32

# Base model – optimal for gFID
python -m torch.distributed.run --master_port 17828 --nproc_per_node=8 sample.py \
    --model DiT1D-B \
    --ckpt pretrained_models/fsdd_base_128_4096.pth \
    --tokenizer_path pretrained_models/set_tokenizer_128_4096.pth \
    --mixed-precision bf16 --force-fp32

# Base model – for high-quality visuals (lower diversity)
python -m torch.distributed.run --master_port 17828 --nproc_per_node=8 sample.py \
    --model DiT1D-B \
    --ckpt pretrained_models/fsdd_base_128_4096.pth \
    --tokenizer_path pretrained_models/set_tokenizer_128_4096.pth \
    --mixed-precision bf16 --force-fp32 \
    --sample_x0 topk --sample_xt topk --top_k 2 --adjust_step 1.0
```

Benchmark Results:
| Model          | rFID ↓ | gFID ↓ |
|----------------|-------|-------|
| DiT Small      | 2.74  | 5.56  |
| DiT Base       | 2.74  | 5.09  |

Note: Experimental results may fluctuate around ±0.1 due to random seed variations.

---

## 🎓 Training Your Own Model

### Preparing Dataset:
Prepare your ImageNet dataset following this structure:
```
data/
├── n01440764/
│   ├── n01440764_18.JPEG
|   └── ...
├── n01443537/
│   ├── n01443537_16.JPEG
|   └── ...
└── ...
```

### Training scripts
Use provided scripts to train image generative models:
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --master_port=12333 train.py \
    --results-dir logs/fsdd_token128_4096_[small/base]/ \
    --model DiT1D-[S/B] \
    --data-path data/ \
    --tokenizer_path pretrained_models/set_tokenizer_128_4096.pth
```

Optionally, you can enable BF16 training to accelerate training speed: `--mixed-precision bf16 --force-fp32`
For tokenizer training, refer directly to [TiTok](https://github.com/bytedance/1d-tokenizer) GitHub Repository. Ensure you adapt the token permutation step before the decoder accordingly.

---

## 🙏 Acknowledgments

This codebase benefits from the excellent prior works:

- [TiTok](https://github.com/bytedance/1d-tokenizer) for tokenizer implementation.
- [DiT](https://github.com/facebookresearch/DiT) for the backbone architecture design and the code of DDP training and sampling.

We would like to sincerely express our gratitude to the outstanding researchers for their powerful contributions and codebases.

---

## 📖 Citation

If you find this project helpful for your research or use it in your own work, please cite our paper:
```bibtex
@article{geng2025tokenset,
  title   = {Tokenize Image as a Set},
  author  = {Zigang Geng, Mengde Xu, Han Hu, Shuyang Gu},
  journal = {arxiv},
  year    = {2025}
}
```

---

## 📬 Contact & Feedback

For questions or feedback, please don't hesitate to reach out:

- **Geng Zigang**: zigang@mail.ustc.edu.cn
- **Hunyuan Research**

---

⭐️ If this repository helped your research, please star 🌟 this repo 👍!