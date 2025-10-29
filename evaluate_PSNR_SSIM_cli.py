### python script to evaluate PSNR and SSIM following evaluate_PSNR_SSIM.m from MPRNet
### results may be slightly different from Matlab script

import argparse
import os
import torch
import kornia

from tqdm import tqdm
from torchvision.io import read_image

def rgb_to_ycbcr(img):
    '''
    This function converts an RGB image to YCbCr color space,
    following the Matlab implementation of rgb2ycbcr and the standard ITU-R BT.601 conversion matrix.
    The result is slightly different from the Matlab implementation due to rounding errors.
    '''
    # Check if input is a single image or a batch of images
    if img.ndim == 3:  # Single image
        img = img.unsqueeze(0)  # Add a batch dimension
       
    if img.dtype == torch.uint8:
        img = img.float()  # Convert to float

    # Normalize if necessary
    if img.max() > 1.0:
        img = img / 255.0

    # Transformation matrix and offset
    T = torch.tensor([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.000],
        [112.000, -93.786, -18.214]
    ], dtype=img.dtype, device=img.device) / 255
    offset = torch.tensor([16, 128, 128], dtype=img.dtype, device=img.device)

    # Prepare output tensor
    ycbcr_img = torch.zeros_like(img)

    # Apply the conversion for each channel
    for p in range(3):
        ycbcr_img[:, p, :, :] = T[p, 0] * img[:, 0, :, :] + T[p, 1] * img[:, 1, :, :] + T[p, 2] * img[:, 2, :, :] + offset[p] / 255

    
    # output dimension is (batch, channel, height, width)
    return ycbcr_img

# Ensure CUDA is available for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(root_dir, gt_root_dirs, datasets, methods):
    for dataset, gt_root_dir in zip(datasets, gt_root_dirs):
        print(dataset)
        for method in methods:
            file_path = os.path.join(root_dir, dataset, method)

            image_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(('.jpg', '.png'))]
            # Use gt_root_dir to construct gt_files paths
            gt_files = []
            for f in image_files:
                base = os.path.basename(f)
                # Get base name without extension and suffix
                name, _ = os.path.splitext(base)
                if dataset == 'gtav':
                    name = name.split('_')[0]
                
                # Try both .png and .jpg for ground truth
                gt_png = os.path.join(gt_root_dir, name + '.png')
                gt_jpg = os.path.join(gt_root_dir, name + '.jpg')
                
                if os.path.exists(gt_png):
                    gt_files.append(gt_png)
                elif os.path.exists(gt_jpg):
                    gt_files.append(gt_jpg)
                else:
                    raise FileNotFoundError(f"Ground truth file not found for {name} (tried both .png and .jpg)")

            total_psnr = 0.0
            total_ssim = 0.0
            img_num = len(image_files)

            for (img_file, gt_file) in tqdm(zip(image_files, gt_files), 0):
                input_img = read_image(img_file).float().unsqueeze(0)
                gt_img = read_image(gt_file).float().unsqueeze(0)

                input_img, gt_img = input_img.to(device), gt_img.to(device)

                # get the Y channel from YCbCr
                input_img, gt_img = rgb_to_ycbcr(input_img)[:,0,:,:], rgb_to_ycbcr(gt_img)[:,0,:,:]

                psnr_val = kornia.metrics.psnr(input_img.unsqueeze(1), gt_img.unsqueeze(1), max_val=1.0)
                total_psnr += psnr_val

                ssim_val = kornia.metrics.ssim(input_img.unsqueeze(1), gt_img.unsqueeze(1), window_size=11, max_val=1.0).mean()
                total_ssim += ssim_val
                # print(psnr_val, ssim_val, img_file)
                

            avg_psnr = total_psnr / img_num
            avg_ssim = total_ssim / img_num
            print(f'For {method}, PSNR: {avg_psnr}, SSIM: {avg_ssim}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM for image results.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of results')
    parser.add_argument('--gt_root_dirs', type=str, nargs='+', required=True, help='List of ground truth directories, one per dataset')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--methods', type=str, nargs='+', required=True, help='List of method names')
    args = parser.parse_args()
    if len(args.gt_root_dirs) != len(args.datasets):
        raise ValueError('The number of gt_root_dirs must match the number of datasets.')
    evaluate(args.root_dir, args.gt_root_dirs, args.datasets, args.methods)