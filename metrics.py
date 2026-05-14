import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from pytorch_fid import fid_score
import torch
import os

# ------------------ IMAGE QUALITY ------------------

def compute_psnr_ssim(originals, protecteds):
    psnr_vals = []
    ssim_vals = []

    for o, p in zip(originals, protecteds):
        o = cv2.imread(o)
        p = cv2.imread(p)

        o = cv2.resize(o, (256, 256))
        p = cv2.resize(p, (256, 256))

        psnr_vals.append(psnr(o, p))
        ssim_vals.append(ssim(o, p, channel_axis=2))

    return np.mean(psnr_vals), np.mean(ssim_vals)


# ------------------ FID ------------------

def compute_fid(original_dir, protected_dir):
    device = torch.device("cpu")
    return fid_score.calculate_fid_given_paths(
        [original_dir, protected_dir],
        batch_size=8,
        device=device,
        dims=2048
    )


# ------------------ SIMILARITY GRAPHS ------------------

def plot_similarity_hist(similarities):
    plt.hist(similarities, bins=20)
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.savefig("results/similarity_hist.png")
    plt.close()


def plot_cdf(similarities):
    sorted_sim = np.sort(similarities)
    cdf = np.arange(len(sorted_sim)) / len(sorted_sim)

    plt.plot(sorted_sim, cdf)
    plt.title("CDF of Similarity")
    plt.xlabel("Similarity")
    plt.ylabel("Probability")
    plt.savefig("results/cdf.png")
    plt.close()


def plot_far_curve(similarities):
    thresholds = np.linspace(0, 1, 100)
    fars = []

    for t in thresholds:
        fars.append(np.mean(similarities > t))

    plt.plot(thresholds, fars)
    plt.title("FAR Curve")
    plt.xlabel("Threshold")
    plt.ylabel("FAR")
    plt.savefig("results/far_curve.png")
    plt.close()