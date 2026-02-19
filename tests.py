import os
from copy import deepcopy

import cv2
import torch
from tqdm import tqdm

from src.layers import convert_wavelet_name, DWT_2D, IDWT_2D
from src.utils import merge_coeffs_into_spatial
from vis_utils import add_title_to_image, create_images_grid, prepare_input_image


def preprocess_input_image(image, device):
    # Scale to range [-1, 1]
    input_image = torch.from_numpy(image) * (2 / 255.) - 1.
    # Add batch dimension and move to device
    input_image = input_image[None, ...].to(device=device)
    # Convert NHWC -> NCHW
    input_image = input_image.permute(0, 3, 1, 2).contiguous()
    return input_image


def postprocess_image(image):
    return (255 * image[0]).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()


def scale_into_range(x, target_range):
    src_range = (x.min(), x.max())
    src_range_size = src_range[1] - src_range[0]
    target_range_size = target_range[1] - target_range[0]
    x = (x - src_range[0]) * target_range_size / src_range_size + target_range[0]
    return x


def prepare_results_grid(grid_wavelets, image, image_path, n_rows, n_cols, device):
    h, w = image.shape[:2]
    input_image = preprocess_input_image(image, device)
    images = []
    errors = {}
    for wavelet in tqdm(grid_wavelets, desc='wavelet'):
        upd_wavelet_name = convert_wavelet_name(wavelet)
        dwt_layer = DWT_2D(upd_wavelet_name, input_shape_HW=(h, w)).to(device=device)
        idwt_layer = IDWT_2D(upd_wavelet_name, input_shape_HW=(h, w)).to(device=device)
        LL, LH, HL, HH = dwt_layer(deepcopy(input_image))
        vis_LL = scale_into_range(LL, (0, 1))
        vis_LH = scale_into_range(LH, (0, 1))
        vis_HL = scale_into_range(HL, (0, 1))
        vis_HH = scale_into_range(HH, (0, 1))
        vis_image = merge_coeffs_into_spatial([vis_LL, vis_LH, vis_HL, vis_HH])
        vis_image = postprocess_image(vis_image)
        vis_image = add_title_to_image(vis_image, wavelet)
        images.append(vis_image)
        restored_image = idwt_layer(LL, LH, HL, HH)
        diff = float(((restored_image - input_image) ** 2).mean().detach().cpu().numpy())
        errors[wavelet] = diff
    grid_image = create_images_grid(images, n_rows=n_rows, n_cols=n_cols)
    image_dir = os.path.join('.', 'results')
    os.makedirs(image_dir, exist_ok=True)
    grid_image_path = os.path.join(image_dir, os.path.split(image_path)[-1])
    status = cv2.imwrite(grid_image_path, grid_image[..., ::-1])
    print(f'Saved image to {grid_image_path}, status={status}')
    print(f'Reconstruction errors: {errors}')


if __name__ == '__main__':
    image_idx = 0
    image, image_path = prepare_input_image(image_idx)
    grid_wavelets = [
        'haar', 'cdf-9/7', 'cdf-5/3', 'dmey', 'sym2'
    ]
    total_size = len(grid_wavelets)
    n_cols = 5
    n_rows = total_size // n_cols
    if n_rows * n_cols > total_size:
        n_rows += 1
    device = torch.device('cuda')
    prepare_results_grid(grid_wavelets, image, image_path, n_rows=n_rows, n_cols=n_cols, device=device)
