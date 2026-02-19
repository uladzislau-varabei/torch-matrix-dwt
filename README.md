# Discrete wavelet transform (DWT) via matrix multiplication in PyTorch

This repository provides implementation of 1D/2D/3D discrete wavelet transform (DWT) via matrix multiplication in PyTorch. 
Operations can run on both: CPU and GPU, explicit backward pass is implemented. 
All the supported wavelets are from `pywavelets` package.

Available wavelets:
* Biorthogonal: `['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']`
* Reverse biorthogonal: `['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']`
* Coiflets: `[coif1-coif17]`
* Symlets: `[sym2-sym20]`
* Daubechies: `[db1-db38]`
* Haar: `haar`
* CDF-9/7: the same as `bior4.4`
* CDF-5/3: the same as `bior2.2`
* “Discrete” FIR approximation of Meyer wavelet: `dmey`

The implementation heavily relies on `WaveCNet` repository https://github.com/LiQiufu/WaveCNet/tree/master/DWT_IDWT.

# Results

...
