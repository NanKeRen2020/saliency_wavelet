
About
=====

Reimplement saliency estimation using a non-parametric vision model of Naila Murray in C++

We detect the salinecy using 2D-haar separable wavelet  and 2D nonseparable wavelet 

including gabor wavelet of another project https://github.com/NanKeRen2020/saliency_gabor.

Tips: You can try other wavelet like symlet wavelet or other symmetry wavelet, but not asymmetry wavelet.

The saliency map result seem not good as symmetry wavelet, for example Daubechies wavelet.  


Environments:
=============

Ubuntu1604  OpenCV3.4.x


Build & Run:
============

cd saliency_wavelet

step1:  g++ -std=c++11 -o saliency_wavelet saliency_wavelet.cpp `pkg-config --cflags --libs opencv`

step2:  ./saliency_wavelet  deep21.bmp  non
 
original image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/saliency_wavelet/deep21.bmp)

result image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/saliency_wavelet/result.png)

sum of result image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/saliency_wavelet/result_sum.png)


Enjoy!!!
========


References
==========

[1] Peng, SL,et al."Construction of two-dimensional compactly supported orthogonal wavelets filters with linear phase".
    ACTA MATHEMATICA SINICA-ENGLISH SERIES 18.No.4(2002):719–726. 

[2] Daubechies, I.: Ten Lectures on Wavelets, CBMS, 61, SIAM, Philadelphia, (1992).