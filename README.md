
About
=====

This project detect the salinecy using 2D-haar separable wavelet and 2D nonseparable wavelet

including gabor wavelet which also implement in my another project https://github.com/NanKeRen2020/saliency_gabor.

Tips: You can try other wavelet like symlet wavelet or other symmetry wavelet.

But the saliency map result of asymmetry wavelet(like Daubechies wavelet) seem not good as symmetry wavelet.  


Environments
=============

Ubuntu1604  OpenCV3.4.x


Build & Run
============

cd saliency_wavelet

step1:  g++ -std=c++11 -o saliency_wavelet saliency_wavelet.cpp `pkg-config --cflags --libs opencv`

step2:  ./saliency_wavelet  deep21.bmp  non

You can check other options in the main function.  
 
original image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/deep21.bmp)

result image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/result.png)

sum of result image

![image](https://github.com/NanKeRen2020/saliency_wavelet/blob/main/result_sum.png)


Enjoy!!!
========


References
==========

[1] Peng, SL,et al."Construction of two-dimensional compactly supported orthogonal wavelets filters with linear phase".
    ACTA MATHEMATICA SINICA-ENGLISH SERIES 18.No.4(2002):719–726. 

[2] Daubechies, I.: Ten Lectures on Wavelets, CBMS, 61, SIAM, Philadelphia, (1992).

[3] Murray, N. , et al. "Saliency estimation using a non-parametric low-level vision model." CVPR 2011 IEEE, 2011.
