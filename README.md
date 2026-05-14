# Facial Diffusion Protection

This project optimizes multi-target facial privacy protection to effectively obfuscate source identities against unauthorized automatic face recognition (AFR) systems. It builds upon diffusion-based adversarial generation techniques and introduces a refined dual-objective adversarial loss function.

## Key Features

- **Dual-Objective Optimization:** The adversarial loss function is enhanced to simultaneously pull the generated image representations toward mixed-target identities while aggressively pushing them away from the source identities.
- **Robust Obfuscation:** Successfully bypasses local face recognition models by significantly lowering similarity to the source and evading identity-dodging metrics.
- **Multi-Target Support:** Allows the selection of multiple target identities simultaneously (e.g., `--target_choices "1, 2, 3"`) to craft highly transferable and resilient adversarial face images.

## Setup

- **Get the Code**
```shell 
git clone https://github.com/202511040-dev/facial-diffusion.git
cd facial-diffusion
```

- **Build the Environment**
We recommend using Anaconda to build the Python environment:
```shell
conda create -n Facial-Privacy-Protection python=3.11
conda activate Facial-Privacy-Protection
pip install -r requirements.txt
```

- **Download Assets**
Download the pre-trained face recognition models and datasets from [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN) and place them in the `assets` folder. The required folder structure is:
```shell
assets/
├── datasets/
├── face_recognition_models/
│   ├── facenet.pth
│   ├── ir152.pth
│   ├── irse50.pth
│   └── mobile_face.pth
├── obfs_target_images/
├── target_images/
└── test_images/
```

## Usage

- **Aligning Images:**
Before starting the protection, ensure that the images in the dataset folder are aligned:
```shell
python align.py
```

- **Running the Protection Pipeline:**
Execute the main script to start the generation and optimization process:
```shell
python main.py --target_choices "1, 2, 3" 
```
### Available Arguments:
- `--source_dir`: Path to the source images folder.
- `--test_dir`: Path to the test images folder for obfuscation.
- `--target_choices`: Comma-separated list of target identities (e.g., `"1, 2, 3"`).
- `--is_obfuscation`: Boolean flag to switch between impersonation and obfuscation.
- `--adv_optim_weight`: Weight for the adversarial optimization loss.

Check `main.py` for a comprehensive list of configurable options.

## Acknowledgments
This repository is an optimized extension of the framework proposed in *Enhancing Facial Privacy Protection via Weakening Diffusion Purification (CVPR 2025)* and utilizes stable diffusion techniques inspired by [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt).
