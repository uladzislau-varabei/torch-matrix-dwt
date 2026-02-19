# Thanks to https://github.com/LiQiufu/WaveCNet/blob/master/DWT_IDWT/DWT_IDWT_layer.py
# Code refactored

import math

import numpy as np
import pywt
import torch
from torch import nn

from src.functional import DWTFunction_1D, IDWTFunction_1D, \
    DWTFunction_2D, IDWTFunction_2D, DWTFunction_2D_tiny, DWTFunction_3D, IDWTFunction_3D


__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D', 'DWT_2D_tiny']

DTYPE = np.float32


def to_module_param(x):
    # Note: probably replace with buffers later
    return nn.Parameter(torch.from_numpy(x), requires_grad=False)


def convert_wavelet_name(name):
    """Converts input wavelet name to the one compatible with pywavelets."""
    return {
        'CDF-9/7'.lower(): 'bior4.4',
        'CDF-5/3'.lower(): 'bior2.2',
        'Haar'.lower(): 'Haar',
        # 'Daubechies-4': 1,
        # 'Coiflet-12': 1,
        'Bior_spline-3/3'.lower(): 'bior3.3',
        'Bior_spline-3/5'.lower(): 'bior3.5',
        'Bior_spline-3/7'.lower(): 'bior3.7',
        'Bior_spline-3/9'.lower(): 'bior3.9',
        # 'Bior_spline-4/8': 1, # Not available? Only 5.5 and 6.8
        'Rev_bior_spline-3/3'.lower(): 'rbio3.3',
        'Rev_bior_spline-3/5'.lower(): 'rbio3.5',
        'Rev_bior_spline-3/7'.lower(): 'rbio3.7',
        'Rev_bior_spline-3/9'.lower(): 'rbio3.9',
        # 'Rev_bior_spline-4/8': 1, # Not available? Only 5.5 and 6.8
    }.get(name.lower(), name.lower())


class DWTBase(nn.Module):
    def __init__(self, wavename, input_shape, forward_mode=False, inverse_mode=False):
        super().__init__()
        assert forward_mode or inverse_mode
        assert not (forward_mode and inverse_mode)
        wavelet = pywt.Wavelet(wavename)
        if forward_mode:
            self.band_low = wavelet.rec_lo
            self.band_high = wavelet.rec_hi
        elif inverse_mode:
            self.band_low = wavelet.dec_lo
            self.band_high = wavelet.dec_hi
            self.band_low.reverse()
            self.band_high.reverse()
        else:
            assert False
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.build_matrices(input_shape)

    def build_matrices(self, input_shape):
        raise NotImplementedError(f'Implement matrices building for {self.__class__.__name__}')

    def forward(self, *args):
        raise NotImplementedError(f'Implement forward for {self.__class__.__name__}')


# ----- 1D transform -----

class DWT_1D(DWTBase):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: L -- (N, C, Length/2)
            H -- (N, C, Length/2)
    """
    def __init__(self, wavename, input_shape):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        """
        super(DWT_1D, self).__init__(wavename, input_shape, forward_mode=True)

    def build_matrices(self, input_shape):
        assert len(input_shape) == 1, f'{self.__class__.__name__}: input_shape={input_shape}'
        self.input_size = input_shape[0]

        L1 = self.input_size
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2

        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]

        self.matrix_low = to_module_param(matrix_h)
        self.matrix_high = to_module_param(matrix_g)

    def forward(self, input):
        """
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        S = input.shape[-1]
        assert self.input_size == S, f'layer_S={self.input_size}, input_S={S}'
        dtype = input.dtype
        return DWTFunction_1D.apply(input, self.matrix_low.to(dtype), self.matrix_high.to(dtype))


class IDWT_1D(DWTBase):
    """
    input:  L -- (N, C, Length/2)
            H -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """
    def __init__(self, wavename, input_shape):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        """
        super(IDWT_1D, self).__init__(wavename, input_shape, inverse_mode=True)

    def build_matrices(self, input_shape):
        assert len(input_shape) == 1, f'{self.__class__.__name__}: input_shape={input_shape}'
        self.input_size = input_shape[0]

        L1 = self.input_size
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2

        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]

        self.matrix_low = to_module_param(matrix_h)
        self.matrix_high = to_module_param(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        S = L.shape[-1] + H.shape[-1]
        assert self.input_size == S, f'layer_S={self.input_size}, input_S={S}'
        dtype = L.dtype
        return IDWTFunction_1D.apply(L, H, self.matrix_low.to(dtype), self.matrix_high.to(dtype))


# ----- Tiny 2D transform -----

class DWT_2D_tiny(DWTBase):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, all four components could be got using DWT_2D
    """
    def __init__(self, wavename, input_shape_HW):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        """
        super(DWT_2D_tiny, self).__init__(wavename, input_shape_HW, forward_mode=True)

    def build_matrices(self, input_shape_HW):
        assert len(input_shape_HW) == 2, f'{self.__class__.__name__}: input_shape={input_shape_HW}'
        self.input_height = input_shape_HW[0]
        self.input_width = input_shape_HW[1]

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = to_module_param(matrix_h_0)
        self.matrix_low_1 = to_module_param(matrix_h_1)
        self.matrix_high_0 = to_module_param(matrix_g_0)
        self.matrix_high_1 = to_module_param(matrix_g_1)

    def forward(self, input):
        """
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        H, W = input.shape[-2], input.shape[-1]
        assert self.input_height == H and self.input_width == W, \
            f'layer_H={self.input_height}, layer_W={self.input_width}, input_H={H}, input_W={W}'
        dtype = input.dtype
        return DWTFunction_2D_tiny.apply(input,
                                         self.matrix_low_0.to(dtype),
                                         self.matrix_low_1.to(dtype),
                                         self.matrix_high_0.to(dtype),
                                         self.matrix_high_1.to(dtype))


# ----- 2D transform -----

class DWT_2D(DWTBase):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
              LH: (N, C, H/2, W/2)
              HL: (N, C, H/2, W/2)
              HH: (N, C, H/2, W/2)
    """
    def __init__(self, wavename, input_shape_HW):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        """
        super(DWT_2D, self).__init__(wavename, input_shape_HW, forward_mode=True)

    def build_matrices(self, input_shape_HW):
        assert len(input_shape_HW) == 2, f'{self.__class__.__name__}: input_shape={input_shape_HW}'
        self.input_height = input_shape_HW[0]
        self.input_width = input_shape_HW[1]

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = to_module_param(matrix_h_0)
        self.matrix_low_1 = to_module_param(matrix_h_1)
        self.matrix_high_0 = to_module_param(matrix_g_0)
        self.matrix_high_1 = to_module_param(matrix_g_1)

    def forward(self, input):
        """
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        H, W = input.shape[-2], input.shape[-1]
        assert self.input_height == H and self.input_width == W, \
            f'layer_H={self.input_height}, layer_W={self.input_width}, input_H={H}, input_W={W}'
        dtype = input.dtype
        return DWTFunction_2D.apply(input,
                                    self.matrix_low_0.to(dtype),
                                    self.matrix_low_1.to(dtype),
                                    self.matrix_high_0.to(dtype),
                                    self.matrix_high_1.to(dtype))


class IDWT_2D(DWTBase):
    """
    input:  LL -- (N, C, H/2, W/2)
            LH -- (N, C, H/2, W/2)
            HL -- (N, C, H/2, W/2)
            HH -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """
    def __init__(self, wavename, input_shape_HW):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        """
        super(IDWT_2D, self).__init__(wavename, input_shape_HW, inverse_mode=True)

    def build_matrices(self, input_shape_HW):
        assert len(input_shape_HW) == 2, f'{self.__class__.__name__}: input_shape={input_shape_HW}'
        self.input_height = input_shape_HW[0]
        self.input_width = input_shape_HW[1]

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        self.matrix_low_0 = to_module_param(matrix_h_0)
        self.matrix_low_1 = to_module_param(matrix_h_1)
        self.matrix_high_0 = to_module_param(matrix_g_0)
        self.matrix_high_1 = to_module_param(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        reconstructing the original 2D data
        :param LL: the low-frequency component
        :param LH: the high-frequency component
        :param HL: the high-frequency component
        :param HH: the high-frequency component
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        H = LL.shape[-2] + HH.shape[-2]
        W = LL.shape[-1] + HH.shape[-1]
        assert self.input_height == H and self.input_width == W, \
            f'layer_H={self.input_height}, layer_W={self.input_width}, input_H={H}, input_W={W}'
        dtype = LL.dtype
        return IDWTFunction_2D.apply(LL, LH, HL, HH,
                                     self.matrix_low_0.to(dtype),
                                     self.matrix_low_1.to(dtype),
                                     self.matrix_high_0.to(dtype),
                                     self.matrix_high_1.to(dtype))


# ----- 3D transform -----

class DWT_3D(DWTBase):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: LLL -- (N, C, D/2, H/2, W/2)
            LLH -- (N, C, D/2, H/2, W/2)
            LHL -- (N, C, D/2, H/2, W/2)
            LHH -- (N, C, D/2, H/2, W/2)
            HLL -- (N, C, D/2, H/2, W/2)
            HLH -- (N, C, D/2, H/2, W/2)
            HHL -- (N, C, D/2, H/2, W/2)
            HHH -- (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename, input_shape_DHW):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        """
        super(DWT_3D, self).__init__(wavename, input_shape_DHW, forward_mode=True)

    def build_matrices(self, input_shape_DHW):
        assert len(input_shape_DHW) == 3, f'{self.__class__.__name__}: input_shape_DHW={input_shape_DHW}'
        self.input_depth = input_shape_DHW[0]
        self.input_height = input_shape_DHW[1]
        self.input_width = input_shape_DHW[2]

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]

        self.matrix_low_0 = to_module_param(matrix_h_0)
        self.matrix_low_1 = to_module_param(matrix_h_1)
        self.matrix_low_2 = to_module_param(matrix_h_2)
        self.matrix_high_0 = to_module_param(matrix_g_0)
        self.matrix_high_1 = to_module_param(matrix_g_1)
        self.matrix_high_2 = to_module_param(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        assert len(input.size()) == 5
        input_D = input.shape[-3]
        input_H = input.shape[-2]
        input_W = input.shape[-1]
        assert self.input_depth == input_D and self.input_height == input_H and self.input_width == input_W, \
            f'layer_D={self.input_depth}, layer_H={self.input_height}, layer_W={self.input_width}, ' \
            f'input_D={input_D}, input_H={input_H}, input_W={input_W}'
        dtype = input.dtype
        return DWTFunction_3D.apply(input,
                                    self.matrix_low_0.to(dtype),
                                    self.matrix_low_1.to(dtype),
                                    self.matrix_low_2.to(dtype),
                                    self.matrix_high_0.to(dtype),
                                    self.matrix_high_1.to(dtype),
                                    self.matrix_high_2.to(dtype))


class IDWT_3D(DWTBase):
    """
    input:  LLL -- (N, C, D/2, H/2, W/2)
            LLH -- (N, C, D/2, H/2, W/2)
            LHL -- (N, C, D/2, H/2, W/2)
            LHH -- (N, C, D/2, H/2, W/2)
            HLL -- (N, C, D/2, H/2, W/2)
            HLH -- (N, C, D/2, H/2, W/2)
            HHL -- (N, C, D/2, H/2, W/2)
            HHH -- (N, C, D/2, H/2, W/2)
    output: the original 3D data -- (N, C, D, H, W)
    """
    def __init__(self, wavename, input_shape_DHW):
        """
        3D inverse DWT (IDWT) for 3D data reconstruction
        """
        super(IDWT_3D, self).__init__(wavename, input_shape_DHW, inverse_mode=True)

    def build_matrices(self, input_shape_DHW):
        assert len(input_shape_DHW) == 3, f'{self.__class__.__name__}: input_shape_DHW={input_shape_DHW}'
        self.input_depth = input_shape_DHW[0]
        self.input_height = input_shape_DHW[1]
        self.input_width = input_shape_DHW[2]

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype=DTYPE)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype=DTYPE)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]

        self.matrix_low_0 = to_module_param(matrix_h_0)
        self.matrix_low_1 = to_module_param(matrix_h_1)
        self.matrix_low_2 = to_module_param(matrix_h_2)
        self.matrix_high_0 = to_module_param(matrix_g_0)
        self.matrix_high_1 = to_module_param(matrix_g_1)
        self.matrix_high_2 = to_module_param(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        """
        :param LLL: the low-frequency component
        :param LLH: the high-frequency component
        :param LHL: the high-frequency component
        :param LHH: the high-frequency component
        :param HLL: the high-frequency component
        :param HLH: the high-frequency component
        :param HHL: the high-frequency component
        :param HHH: the high-frequency component
        :return: the original 3D input data
        """
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        input_D = LLL.shape[-3] + HHH.shape[-3]
        input_H = LLL.shape[-2] + HHH.shape[-2]
        input_W = LLL.shape[-1] + HHH.shape[-1]
        assert self.input_depth == input_D and self.input_height == input_H and self.input_width == input_W, \
            f'layer_D={self.input_depth}, layer_H={self.input_height}, layer_W={self.input_width}, ' \
            f'input_D={input_D}, input_H={input_H}, input_W={input_W}'
        dtype = LLL.dtype
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0.to(dtype),
                                     self.matrix_low_1.to(dtype),
                                     self.matrix_low_2.to(dtype),
                                     self.matrix_high_0.to(dtype),
                                     self.matrix_high_1.to(dtype),
                                     self.matrix_high_2.to(dtype))
