import numpy as np
import torch


NCHW_FORMAT = 'NCHW'
NHWC_FORMAT = 'NHWC'
to_NHWC_axis = [0, 2, 3, 1] # NCHW -> NHWC
to_NCHW_axis = [0, 3, 1, 2] # NHWC -> NCHW

# Note: in PyTorch NCHW and NHWC are the same as for indexes, only strides differ, so always use NCHW
DEFAULT_DATA_FORMAT = NCHW_FORMAT


def get_default_coeffs_scales_2d(COEFFS_SCALES_V):
    # 2d transform, so power 2
    COEFFS_SCALES_2D_v1 = np.array([
        1 / np.sqrt(2),
        np.sqrt(2),
        np.sqrt(2),
        np.sqrt(2)
    ], dtype=np.float32) ** 2

    # The same scales allows to get coeffs ranges that are consistent
    COEFFS_SCALES_2D_v2 = np.array([
        1 / np.sqrt(2) ** 2,
        1 / np.sqrt(2) ** 2,
        1 / np.sqrt(2) ** 2,
        1 / np.sqrt(2) ** 2
    ], dtype=np.float32)

    # 2d transform, so use double power only for LL coeffs
    COEFFS_SCALES_2D_v3 = np.array([
        1 / np.sqrt(2) ** 2,
        1 / np.sqrt(2),
        1 / np.sqrt(2),
        1 / np.sqrt(2)
    ], dtype=np.float32)

    COEFFS_SCALES_2D_v4 = np.array([
        1 / np.sqrt(2),
        1,
        1,
        1
    ], dtype=np.float32)

    COEFFS_SCALES_2D_v5 = np.array([
        1 / np.sqrt(2),
        1,
        1,
        np.sqrt(2)
    ], dtype=np.float32)

    # LL taken from v3, H coeffs from v5
    COEFFS_SCALES_2D_v6 = np.array([
        1 / np.sqrt(2) ** 2,
        1,
        1,
        np.sqrt(2)
    ], dtype=np.float32)

    COEFFS_SCALES_2D_DICT = {
        1: COEFFS_SCALES_2D_v1,
        2: COEFFS_SCALES_2D_v2,
        3: COEFFS_SCALES_2D_v3,
        4: COEFFS_SCALES_2D_v4,
        5: COEFFS_SCALES_2D_v5,
        6: COEFFS_SCALES_2D_v6
    }

    COEFFS_SCALES_2D = torch.from_numpy(COEFFS_SCALES_2D_DICT[COEFFS_SCALES_V])
    return COEFFS_SCALES_2D

# 6 is the best for preserving source data range for LL and keeping similar ranges for all H details
# Found with tests.py with and without normalization for input
COEFFS_SCALES_V = 6
COEFFS_SCALES_2D = get_default_coeffs_scales_2d(COEFFS_SCALES_V)

DEFAULT_SCALE_2D_COEFFS = True

# Scales for LL, LH, HL, HH after DWT. Before IDWT inverse values must be used
# Note: probably it's better to combine scaled init with these coeffs.
# In this case coeffs can be scaled by corresponding values, e.g.,
# layer_scales = [2, 16, 16, 24] and weight_init_scales = [1, 8, 8, 8]
LAYER_COEFFS_SCALES = [2, 48, 48, 64]


# ----- Merge/extract utils -----

def merge_coeffs_into_channels(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
    x_LL, x_LH, x_HL, x_HH = x_coeffs
    if data_format == NCHW_FORMAT:
        concat_axis = 1
    else: # if data_format == NHWC_FORMAT:
        concat_axis = 3
    return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=concat_axis)


def extract_coeffs_from_channels(x, data_format=DEFAULT_DATA_FORMAT):
    if data_format == NCHW_FORMAT:
        n = x.shape[1] // 4
        x_LL = x[:, (0 * n) : (1 * n), :, :]
        x_LH = x[:, (1 * n) : (2 * n), :, :]
        x_HL = x[:, (2 * n) : (3 * n), :, :]
        x_HH = x[:, (3 * n) : (4 * n), :, :]
    else: # if data_format == NHWC_FORMAT:
        n = x.shape[3] // 4
        x_LL = x[:, :, :, (0 * n) : (1 * n)]
        x_LH = x[:, : ,:, (1 * n) : (2 * n)]
        x_HL = x[:, :, :, (2 * n) : (3 * n)]
        x_HH = x[:, :, :, (3 * n) : (4 * n)]
    return x_LL, x_LH, x_HL, x_HH


def merge_coeffs_into_spatial(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
    x_LL, x_LH, x_HL, x_HH = x_coeffs
    if data_format == NCHW_FORMAT:
        h_axis, v_axis = 2, 3
    else:  # if data_format == NHWC_FORMAT:
        h_axis, v_axis = 1, 2
    x = torch.cat([
        torch.cat([x_LL, x_LH], dim=h_axis),
        torch.cat([x_HL, x_HH], dim=h_axis)
    ], dim=v_axis)
    return x


def extract_coeffs_from_spatial(x, data_format=DEFAULT_DATA_FORMAT):
    if data_format == NCHW_FORMAT:
        _, C, H, W = x.shape
        x_LL = x[:, :, : H // 2, : W // 2]
        x_LH = x[:, :, H // 2 :, : W // 2]
        x_HL = x[:, :, : H // 2, W // 2: ]
        x_HH = x[:, :, H // 2 :, W // 2: ]
    else:  # data_format == NHWC_FORMAT:
        _, H, W, C = x.shape
        x_LL = x[:, : H // 2, : W // 2, :]
        x_LH = x[:, H // 2 :, : W // 2, :]
        x_HL = x[:, : H // 2, W // 2 :, :]
        x_HH = x[:, H // 2 :, W // 2 :, :]
    return x_LL, x_LH, x_HL, x_HH