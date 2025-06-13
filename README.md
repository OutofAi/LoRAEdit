# LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning

[[Paper]](https://arxiv.org/pdf/2506.10082) | [[Project Page]](https://cjeen.github.io/LoraEditPaper/)

<div align="center">
  <img src="assets/figs_teaser.png" alt="LoRA-Edit Teaser" width="800"/>
</div>
We achieves high-quality first-frame guided video editing given a reference image (top row), while maintaining flexibility for incorporating additional reference conditions (bottom row).

## 📰 News

- **[2025.06.07]** LoRA-Edit first-frame-guided-editing code is now available! 🎉

## 🛠️ Environment Setup

### Prerequisites
- CUDA-compatible GPU with sufficient VRAM (We use a single GeForce RTX 4090 (24GB))
- Python 3.12 (recommended)
- Git
- Miniconda or Anaconda

### 1. Clone Repository and Setup Environment

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/cjeen/LoRAEdit.git
cd LoRAEdit

# If you already cloned without submodules, run:
# git submodule init
# git submodule update
```

### 2. Install Miniconda (if not already installed)

Download and install from: https://docs.anaconda.com/miniconda/

### 3. Create Conda Environment

```bash
# Create environment with Python 3.12
conda create -n lora-edit python=3.12
conda activate lora-edit
```

### 4. Install PyTorch

**Important**: Install PyTorch 2.6.0 with CUDA 12.4 for flash attention compatibility:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 5. Install NVCC

Install nvcc to match your CUDA version: https://anaconda.org/nvidia/cuda-nvcc

```bash
conda install -c nvidia cuda-nvcc
```

### 6. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 7. Install SAM2

```bash
# Clone and install SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

### 8. Download Models

#### Download Wan2.1-I2V Model
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the Wan2.1-I2V model
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
```

#### Download SAM2 Model Checkpoint
```bash
# Create models directory
mkdir -p models_sam

# Download SAM2 large model (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O models_sam/sam2_hiera_large.pt

# Alternative: Download other SAM2 models if needed
# SAM2 Base+: wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -O models_sam/sam2_hiera_base_plus.pt
# SAM2 Small: wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -O models_sam/sam2_hiera_small.pt
# SAM2 Tiny: wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -O models_sam/sam2_hiera_tiny.pt
```

## 🚀 Usage
### Tutorial Video

Watch this quick tutorial to see how to use the data preprocessing interface:

https://github.com/user-attachments/assets/a03ee16a-c816-4284-8f45-a3cbbed4c702

### Step 1: Data Preprocessing

Launch the data preprocessing interface:

```bash
python predata_app.py --port 8890 --checkpoint_dir models_sam/sam2_hiera_large.pt
```

### Step 2: LoRA Training

After preprocessing, use the generated training command (example):

```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config ./processed_data/your_sequence/configs/training.toml
```

### Step 3: Video Generation

After training completes, run inference:

```bash
# Save your edited first frame as edited_image.png (or .jpg) in the data directory
# Then run inference
python inference.py --model_root_dir ./Wan2.1-I2V-14B-480P --data_dir ./processed_data/your_sequence
```

## 📁 Directory Structure

```
project_root/
├── predata_app.py          # Data preprocessing interface
├── train.py                # LoRA training script
├── inference.py            # Video generation inference
├── models_sam/             # SAM2 model checkpoints
│   └── sam2_hiera_large.pt
├── Wan2.1-I2V-14B-480P/    # Wan2.1 model directory
├── processed_data/         # Processed training data
│   └── your_sequence/
│       ├── traindata/      # Training videos and captions
│       ├── configs/        # Training configuration files
│       ├── lora/          # Trained LoRA checkpoints
│       ├── inference_rgb.mp4    # Preprocessed RGB video
│       ├── inference_mask.mp4   # Mask video
│       └── edited_image.png     # Your edited first frame
└── requirements.txt
```

## 🙏 Acknowledgments

This project is built upon [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) by tdrussell. We gratefully acknowledge their excellent work in providing a solid foundation for pipeline parallel training of diffusion models.

The SAM2 GUI interface in this project references code from [SAM2-GUI](https://github.com/YunxuanMao/SAM2-GUI) by YunxuanMao. We thank them for their contribution to the SAM2 community with their intuitive interface design.
