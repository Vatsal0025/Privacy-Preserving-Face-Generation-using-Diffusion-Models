# =============================================================================
# Import required libraries
# =============================================================================
import os
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import *

from metrics import (
    compute_psnr_ssim,
    compute_fid,
    plot_similarity_hist,
    plot_cdf,
    plot_far_curve
)


def attack_local_models(args, protection):
    all_similarities = []
    original_paths = []
    protected_paths = []
    # Load test model
    test_model = load_FR_models(args, args.test_model_name)
    # False acceptance rate (FAR) is set to 0.01
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922),
               'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191),
               'mobile_face': (0.183635, 0.301611, 0.380878)}

    target_folder = "_".join(args.target_choices)
    result_dir = args.protected_image_dir + '/' + \
        args.test_model_name[0] + '/' + target_folder
    result_fn = os.path.join(result_dir, "result.txt")

    print('Protection:', protection)
    with open(result_fn, 'a') as f:
        f.write(f"Protection: {protection}\n")
    f.close()

    combined_dir = os.path.join(result_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    size = test_model[args.test_model_name[0]][0]
    model = test_model[args.test_model_name[0]][1]
    #
    # _, test_image = get_target_test_images(args.target_choice,
    #                                        args.device,
    #                                        args.MTCNN_cropping)
    # test_embbeding = model(
    #     (F.interpolate(test_image, size=size, mode='bilinear')))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ADDED
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    target_images = []
    for c in args.target_choices:
        img, _ = get_target_test_images(c, args.device, args.MTCNN_cropping)
        target_images.append(img)

    target_embeddings_list = [
        model(F.interpolate(img, size=size, mode='bilinear'))
        for img in target_images
    ]

    # fusion embedding
    fusion_embedding = torch.mean(torch.stack(target_embeddings_list), dim=0)
    fusion_embedding = fusion_embedding / fusion_embedding.norm(dim=1, keepdim=True)


    FAR01 = 0
    FAR001 = 0
    FAR0001 = 0
    total = 0
    if protection:
        for img_path in glob.glob(os.path.join(result_dir, "*.png")):
            protectec_image = read_img(img_path, 0.5, 0.5, args.device)
            if args.MTCNN_cropping:
                bb_src1 = alignment(Image.open(img_path).convert("RGB"))
                protected_image_hold = protectec_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                #
                _, _, h, w = protected_image_hold.shape
                if h != 0 and w != 0:
                    protectec_image = protected_image_hold
            ae_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            # cos_simi = torch.cosine_similarity(ae_embbeding, test_embbeding)
            # ----------------------------------------------------------------------
            # ----------------------------------------------------------------------
            # ADDED
            # ----------------------------------------------------------------------
            # ----------------------------------------------------------------------
            cos_simi = torch.cosine_similarity(ae_embbeding, fusion_embedding)
            
            all_similarities.append(cos_simi.item())
            fn = img_path.split("\\")[-1].split(".")[0] + ".png"
            original_paths.append(os.path.join(args.source_dir, fn))
            protected_paths.append(img_path)

            if cos_simi.item() > th_dict[args.test_model_name[0]][0]:
                FAR01 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][1]:
                FAR001 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][2]:
                FAR0001 += 1
            total += 1

            # Combine the clean and protected images for visualization
            adv_img = cv2.imread(img_path)
            fn = img_path.split("\\")[-1].split(".")[0] + ".png"
            clean_img = cv2.imread(os.path.join(args.source_dir, fn))
            if clean_img.shape[0] != args.image_size:
                clean_img = cv2.resize(clean_img, (args.image_size, args.image_size),
                                       interpolation=cv2.INTER_LANCZOS4)
            #
            combined_img = np.concatenate([clean_img, adv_img], 1)
            combined_fn = f"{fn.split('.')[0]}_cos_simi_{cos_simi.item():.4f}.png"
            cv2.imwrite(os.path.join(combined_dir, combined_fn), combined_img)
    else:
        for img in tqdm(os.listdir(args.source_dir), desc=args.test_model_name[0] + ' clean'):
            protectec_image = read_img(os.path.join(
                args.source_dir, img), 0.5, 0.5, args.device)
            if args.MTCNN_cropping:
                bb_src1 = alignment(Image.open(os.path.join(
                    args.source_dir, img)))
                protected_image_hold = protectec_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                #
                _, _, h, w = protected_image_hold.shape
                if h != 0 and w != 0:
                    protectec_image = protected_image_hold
            ae_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            # cos_simi = torch.cosine_similarity(ae_embbeding, test_embbeding)
            # ----------------------------------------------------------------------
            # ----------------------------------------------------------------------
            # ADDED
            # ----------------------------------------------------------------------
            # ----------------------------------------------------------------------
            cos_simi = torch.cosine_similarity(ae_embbeding, fusion_embedding)

            all_similarities.append(cos_simi.item())
            original_paths.append(os.path.join(args.source_dir, img))
            protected_paths.append(os.path.join(args.source_dir, img))
            
            if cos_simi.item() > th_dict[args.test_model_name[0]][0]:
                FAR01 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][1]:
                FAR001 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][2]:
                FAR0001 += 1
            total += 1
    
    os.makedirs("results", exist_ok=True)
    np.save("results/similarity.npy", np.array(all_similarities))

    result_str = f"{args.test_model_name[0]} PSR in FAR@0.1: {FAR01/total:.4f}, PSR in FAR@0.01: {FAR001/total:.4f}, PSR in FAR@0.001: {FAR0001/total:.4f}\n"
    print(result_str)
    with open(result_fn, 'a') as f:
        f.write(result_str)
    
    plot_similarity_hist(all_similarities)
    plot_cdf(all_similarities)
    plot_far_curve(all_similarities)

    psnr_val, ssim_val = compute_psnr_ssim(original_paths, protected_paths)
    protected_dir = result_dir if protection else args.source_dir
    fid_val = compute_fid(args.source_dir, protected_dir)

    metrics_str = f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, FID: {fid_val:.4f}\n"
    print(metrics_str)
    with open(result_fn, 'a') as f:
        f.write(metrics_str)