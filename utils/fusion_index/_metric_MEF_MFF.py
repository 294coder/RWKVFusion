# --------------------------------------------------------------------------
# Information and Copyright

# Author: Zihan Cao
# Date: 2024-10-01
# GPL-3.0 License
# Copyright (c) 2024 by Zihan Cao and UESTC, All rights reserved.

# --------------------------------------------------------------------------

# Code modified from MEFB, thanks for the authors' sharing
# Reference: https://github.com/xingchenzhang/MEFB

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------

# Contact:
#   Name: Zihan Cao
#   Email: iamzihan666@gmail.com
#   Affiliation: School of Mathematics, University of Electronic Science and Technology of China (UESTC)

# --------------------------------------------------------------------------

import math
from typing import List, Optional, Sequence, Tuple, Union

import brisque
import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from pyiqa import create_metric
from torch import Tensor
from torchmetrics.functional.image.ssim import _ssim_update
from torchmetrics.functional.image.utils import (
    _gaussian_kernel_2d,
    _gaussian_kernel_3d,
    _reflection_pad_3d,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment


def safe_mean(tensor):
    # Create a mask to mark elements that are not NaN and not Inf
    valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)

    # Replace invalid values with 0
    valid_values = torch.where(
        valid_mask, tensor, torch.tensor(0.0, device=tensor.device)
    )

    # Calculate the sum and count of valid values
    sum_valid = valid_values.sum()
    count_valid = valid_mask.sum()

    # Calculate the mean of valid values
    mean_valid = torch.where(
        count_valid > 0,
        sum_valid / count_valid,
        torch.tensor(float("nan"), device=tensor.device),
    )

    return mean_valid


def CrossEN(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    """
    ```matlab
    % The source code is from the authors of the metric
    % The interface is modified by the author of MEFB to integrate it into MEFB.
    %
    % Reference for the metric:
    % D. M. Bulanon, T. Burks, and V. Alchanatis, ?��Image fusion of visible and thermal images for fruit detection,?��
    % Biosystems Engineering, vol. 103, no. 1, pp. 12��C22, 2009.

    function res=metricsCross_entropy(img1,img2,fused)

        % Get the size of img
        [m,n,b] = size(fused);
        [m1,n1,b1] = size(img1);
        [m2,n2,b2] = size(img2);

        if (b1 == 1) && (b2 ==1) && (b == 3)
            fused_new = zeros(m,n);
            fused_new = fused(:,:,1);
            fused = fused_new;
        end
        [m,n,b] = size(fused);

        if b == 1
            g = cross_entropy_single(img1,img2,fused);
            res = g;
        elseif b1 == 1
            for k = 1 : b
            g(k) = cross_entropy_single(img1(:,:,k), img2,fused(:,:,k));
            end
            res = mean(g);
        else
            for k = 1 : b
                g(k) = cross_entropy_single(img1(:,:,k), img2(:,:,k),fused(:,:,k));
            end
            res = mean(g);
        end

    end

    function output = cross_entropy_single(img1,img2,fused)

        cross_entropyVI = cross_entropy(img1,fused);
        cross_entropyIR = cross_entropy(img2,fused);
        output = (cross_entropyVI + cross_entropyIR)./2.0;

    end

    function res0 = cross_entropy(img1,fused)
        s=size(size(img1));
        if s(2)==3
            f1=rgb2gray(img1);
        else
            f1=img1;
        end

        s1=size(size(fused));
        if s1(2)==3
            f2=rgb2gray(fused);
        else
            f2=fused;
        end

        G1=double(f1);
        G2=double(f2);
        [m1,n1]=size(G1);
        [m2,n2]=size(G2);
        m2=m1;
        n2=n1;
        X1=zeros(1,256);
        X2=zeros(1,256);
        result=0;

        for i=1:m1
            for j=1:n1
                X1(G1(i,j)+1)=X1(G1(i,j)+1)+1;
                X2(G2(i,j)+1)=X2(G2(i,j)+1)+1;
            end
        end

        for k=1:256
            P1(k)=X1(k)/(m1*n1);
            P2(k)=X2(k)/(m1*n1);
            if((P1(k)~=0)&(P2(k)~=0))
                result=P1(k)*log2(P1(k)/P2(k))+result;
            end
        end
        res0=result;
    end
    ```
    """
    count_A = (
        torch.bincount(img_A.flatten().round().long(), minlength=256) / img_A.numel()
    )
    count_B = (
        torch.bincount(img_B.flatten().round().long(), minlength=256) / img_B.numel()
    )
    count_fused = (
        torch.bincount(img_fused.flatten().round().long(), minlength=256)
        / img_fused.numel()
    )

    EN_A = (count_A * torch.log2(count_A / (count_fused + (count_fused == 0)))).sum()
    EN_B = (count_B * torch.log2(count_B / (count_fused + (count_fused == 0)))).sum()

    return (EN_A + EN_B) / 2

    # def cross_entropy_single(img1, img2, fused):
    #     cross_entropyVI = cross_entropy(img1, fused)
    #     cross_entropyIR = cross_entropy(img2, fused)
    #     return (cross_entropyVI + cross_entropyIR) / 2.0

    # def cross_entropy(img1, fused):
    #     G1 = img1.float()
    #     G2 = fused.float()

    #     X1 = torch.histc(G1, bins=256, min=0, max=255)
    #     X2 = torch.histc(G2, bins=256, min=0, max=255)

    #     P1 = X1 / (G1.shape[0] * G1.shape[1])
    #     P2 = X2 / (G2.shape[0] * G2.shape[1])

    #     mask = (P1 != 0) & (P2 != 0)
    #     result = torch.sum(P1[mask] * torch.log2(P1[mask] / P2[mask]))

    #     return result

    # return cross_entropy_single(img_A, img_B, img_fused)


def _ssim_update(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Structural Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the contrast term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    """
    is_3d = preds.ndim == 5

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if len(kernel_size) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(kernel_size) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )
    if len(sigma) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(sigma) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )

    if return_full_image and return_contrast_sensitivity:
        raise ValueError(
            "Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(
            f"Expected `kernel_size` to have odd positive number. Got {kernel_size}."
        )

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())  # type: ignore[call-overload]
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]

    c1 = pow(k1 * data_range, 2)  # type: ignore[operator]
    c2 = pow(k2 * data_range, 2)  # type: ignore[operator]
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    pad_h = (gauss_kernel_size[0] - 1) // 2
    pad_w = (gauss_kernel_size[1] - 1) // 2

    if is_3d:
        pad_d = (gauss_kernel_size[2] - 1) // 2
        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
        if gaussian_kernel:
            kernel = _gaussian_kernel_3d(
                channel, gauss_kernel_size, sigma, dtype, device
            )
    else:
        preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if gaussian_kernel:
            kernel = _gaussian_kernel_2d(
                channel, gauss_kernel_size, sigma, dtype, device
            )

    if not gaussian_kernel:
        kernel = torch.ones(
            (channel, 1, *kernel_size), dtype=dtype, device=device
        ) / torch.prod(torch.tensor(kernel_size, dtype=dtype, device=device))

    input_list = torch.cat(
        (preds, target, preds * preds, target * target, preds * target)
    )  # (5 * B, C, H, W)

    outputs = (
        F.conv3d(input_list, kernel, groups=channel)
        if is_3d
        else F.conv2d(input_list, kernel, groups=channel)
    )

    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    # Calculate the variance of the predicted and target images, should be non-negative
    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target.to(dtype) + c2
    lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / (
        (mu_pred_sq + mu_target_sq + c1) * lower
    )

    if is_3d:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
    else:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]

    if return_contrast_sensitivity:
        contrast_sensitivity = upper / lower
        if is_3d:
            contrast_sensitivity = contrast_sensitivity[
                ..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d
            ]
        else:
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w]
        return (
            ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1),
            sigma_pred_target,
            contrast_sensitivity.reshape(contrast_sensitivity.shape[0], -1).mean(-1),
        )

    if return_full_image:
        return (
            ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1),
            sigma_pred_target,
            ssim_idx_full_image,
        )

    return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), sigma_pred_target


def Qc(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    """
    ```matlab
    % The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricCvejic.m
    % The interface is modified by the author of MEFB
    %
    % Reference for the metric
    % N. Cvejic, A. Loza, D. Bull, N. Canagarajah, A similarity metric for assessment of image fusion
    % algorithms, Int. J. Signal Process. 2 (3) (2005) 178�C182.

    function res = metricsQc(img1,img2,fused)

        % Get the size of img
        [m,n,b] = size(fused);
        [m1,n1,b1] = size(img1);
        [m2,n2,b2] = size(img2);

        if (b1 == 1) && (b2 ==1) && (b == 3)
            fused_new = zeros(m,n);
            fused_new = fused(:,:,1);
            fused = fused_new;
        end
        [m,n,b] = size(fused);

        if b == 1
            g = Qc(img1,img2,fused);
            res = g;
        elseif b1 == 1
            for k = 1 : b
                g(k) = Qc(img1(:,:,k),img2,fused(:,:,k));
            end
            res = mean(g);
        else
            for k = 1 : b
                g(k) = Qc(img1(:,:,k),img2(:,:,k),fused(:,:,k));
            end
            res = mean(g);
        end
    end


    function output = Qc(im1,im2,fused)

    % function res=metricCvejic(im1,im2,fused,sw)
    %
    % This function implements Chen's algorithm for fusion metric.
    % im1, im2 -- input images;
    % fused      -- fused image;
    % res      -- metric value;
    % sw       -- 1: metric 1; 2: metric 2. Cvejic has two different metics.
    %
    % IMPORTANT: The size of the images need to be 2X.
    %
    % Z. Liu [July 2009]
    %

    % Ref: Metric for multimodal image sensor fusion, Electronics Letters, 43 (2) 2007
    % by N. Cvejic et al.
    %
    % Ref: A Similarity Metric for Assessment of Image Fusion Algorithms, International Journal of Information and Communication Engineering 2 (3) 2006, pp.178-182.
    % by N. Cvejic et al.
    %


    %% pre-processing
        s=size(size(im1));
        if s(2)==3
            im1=rgb2gray(im1);
        else
            im1=im1;
        end

        s1=size(size(im2));
        if s1(2)==3
            im2=rgb2gray(im2);
        else
            im2=im2;
        end

        s2=size(size(fused));
        if s2(2)==3
            fused=rgb2gray(fused);
        else
            fused=fused;
        end

        im1=double(im1);
        im2=double(im2);
        fused=double(fused);


        [mssim2, ssim_map2, sigma_XF] = ssim_yang(im1, fused);
        [mssim3, ssim_map3, sigma_YF] = ssim_yang(im2, fused);

        simXYF=sigma_XF./(sigma_XF+sigma_YF);
        sim=simXYF.*ssim_map2+(1-simXYF).*ssim_map3;


        index=find(simXYF<0);
        sim(index)=0;

        index=find(simXYF>1);
        sim(index)=1;

        sim=sim(~isnan(sim));

        output=mean2(sim);

    end

    %%
    function [mssim, ssim_map, sigma12] = ssim_yang(img1, img2)

    %========================================================================

    [M N] = size(img1);
    if ((M < 11) | (N < 11))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return
    end
    window = fspecial('gaussian', 7, 1.5);	%

    L = 255;                                  %

    C1 = 2e-16;
    C2 = 2e-16;

    window = window/sum(sum(window));
    img1 = double(img1);
    img2 = double(img2);
    mu1   = filter2(window, img1, 'valid');
    mu2   = filter2(window, img2, 'valid');
    mu1_sq = mu1.*mu1;
    mu2_sq = mu2.*mu2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
    sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
    if (C1 > 0 & C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
    else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));

    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
    end

    mssim = mean2(ssim_map);

    %return
    end
    """

    def _check_type_dim(img):
        _pad_dim = 4 - img.ndim
        if _pad_dim > 0:
            padded_dim = [None for _ in range(_pad_dim)]
            img = img[*padded_dim]
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32) #/ 255.

        return img.type(torch.float32)

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    ssim_index_A, sigma12_A, ssim_map_A = _ssim_update(
        img_A,
        img_fused,
        kernel_size=7,
        sigma=1.5,
        data_range=(0, 255.0),
        return_full_image=True,
        return_contrast_sensitivity=False,
    )

    ssim_index_B, sigma12_B, ssim_map_B = _ssim_update(
        img_B,
        img_fused,
        kernel_size=7,
        sigma=1.5,
        data_range=(0, 255.0),
        return_full_image=True,
        return_contrast_sensitivity=False,
    )

    simXYF = sigma12_A / (sigma12_A + sigma12_B)
    sim = simXYF * ssim_map_A + (1 - simXYF) * ssim_map_B

    index = torch.where(simXYF < 0, torch.zeros_like(simXYF), sim)
    index = torch.where(simXYF > 1, torch.ones_like(simXYF), sim)
    # index = index[~torch.isnan(index)]
    # index = index[~torch.isinf(index)]

    # return index.mean()
    return safe_mean(index)


def Qp(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    """
    ```matlab
    % The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricZhao.m
    % and https://github.com/zhengliu6699/imageFusionMetrics/blob/master/myphasecong3.m
    % The interface is modified by the authors of MEFB
    % References for the metric
    %
    % J. Zhao, R. Laganiere, Z. Liu, Performance assessment of combinative pixellevel image fusion based on an
    % absolute feature measurement, Int. J. Innovative Comput. Inf. Control 3 (6) (2007) 1433�C1447.

    function res = metricsQp(img1,img2,fused)

        [m,n,b] = size(fused);
        [m1,n1,b1] = size(img1);
        [m2,n2,b2] = size(img2);

        if (b1 == 1) && (b2 ==1) && (b == 3)
            fused_new = zeros(m,n);
            fused_new = fused(:,:,1);
            fused = fused_new;
        end
        [m,n,b] = size(fused);

        if b == 1
            g = Qp(img1,img2,fused);
            res = g;
        elseif b1 == 1
            for k = 1 : b
                g(k) = Qp(img1(:,:,k),img2,fused(:,:,k));
            end
            res = mean(g);
        else
            for k = 1 : b
                g(k) = Qp(img1(:,:,k),img2(:,:,k),fused(:,:,k));
            end
            res = mean(g);
        end
    end

    function output = Qp(im1,im2,fused)

    % function res=pc_assessFusion(im1,im2,fused)
    %
    % This function is to do the assessment for the fused image.
    %
    % im1 ---- the input image one
    % im2 ---- the input image two
    % fused ---- the fused image
    % res ==== the assessment result
    %
    % Z. Liu @NRCC

    % Ref: Performance assessment of combinative pixel-level image fusion based on an absolute feature measurement, International Journal of Innovative Computing, Information and Control, 3 (6A) 2007, pp.1433-1447
    % by J. Zhao et al.
    %

    % some global parameters

    fea_threshold=0.1;  % threshold value for the feature

    % 1) first, calculate the PC

    im1=double(im1);
    im2=double(im2);
    fused=double(fused);

        s=size(size(im1));
        if s(2)==3
            im1=rgb2gray(im1);
        else
            im1=im1;
        end

        s1=size(size(im2));
        if s1(2)==3
            im2=rgb2gray(im2);
        else
            im2=im2;
        end

        s2=size(size(fused));
        if s2(2)==3
            fused=rgb2gray(fused);
        else
            fused=fused;
        end

        im1=double(im1);
        im2=double(im2);
        fused=double(fused);


    [pc1,or1,M1,m1]=myphasecong3(im1);
    clear or1;

    [pc2,or2,M2,m2]=myphasecong3(im2);
    clear or2;

    [pcf,orf,Mf,mf]=myphasecong3(fused);
    clear orf;

    % 2)
    [hang,lie]=size(fused);

    mask=(pc1>pc2);
    pc_max=mask.*pc1+(~mask).*pc2;
    M_max=mask.*M1+(~mask).*M2;
    m_max=mask.*m1+(~mask).*m2;

    mask1=(pc1>fea_threshold);
    mask2=(pc2>fea_threshold);
    mask3=(pc_max>fea_threshold);


    % the PC component
    resultPC=correlation_coeffcient(pc1,pc2,pc_max,pcf,mask1,mask2,mask3);
    clear pc1;
    clear pc2;
    clear pc_max;
    clear pcf;

    resultM=correlation_coeffcient(M1,M2,M_max,Mf,mask1,mask2,mask3);
    clear M1;
    clear M2;
    clear M_max;
    clear Mf;

    resultm=correlation_coeffcient(m1,m2,m_max,mf,mask1,mask2,mask3);
    clear m1;
    clear m2;
    clear m_max;
    clear mf;

    [resultPC resultM resultm]';

    output=resultPC*resultM*resultm;
    end


    %=================================================
    %
    % This sub-function is to calculate the correlation coefficients
    %
    %=================================================

    function res=correlation_coeffcient(im1,im2,im_max,imf, mask1,mask2,mask3)

    % im1, im2, im_max, imf --- the input feature maps
    % mask1~3 --- the corresponding PC map mask for original image 1, 2, max.
    %
    %
    %

    % some local constant parameters
    window=fspecial('gaussian',11,1.5);
    window=window./(sum(window(:)));

    C1=0.0001;
    C2=0.0001;
    C3=0.0001;

    %
    im1=mask1.*im1;
    im2=mask2.*im2;
    im_max=mask3.*im_max;

    mu1=filter2(window,im1,'same');
    mu2=filter2(window,im2,'same');
    muf=filter2(window,imf,'same');
    mu_max=filter2(window,im_max,'same');

    mu1_sq=mu1.*mu1;
    mu2_sq=mu2.*mu2;
    muf_sq=muf.*muf;
    mu_max_sq=mu_max.*mu_max;

    mu1_muf=mu1.*muf;
    mu2_muf=mu2.*muf;
    mu_max_muf=mu_max.*muf;

    sigma1_sq=filter2(window,im1.*im1,'same')-mu1_sq;
    sigma2_sq=filter2(window,im2.*im2,'same')-mu2_sq;
    sigmaMax_sq=filter2(window,im_max.*im_max,'same')-mu_max_sq;
    sigmaf_sq=filter2(window,imf.*imf,'same')-muf_sq;

    sigma1f=filter2(window,im1.*imf,'same')-mu1_muf;
    sigma2f=filter2(window,im2.*imf,'same')-mu2_muf;
    sigmaMaxf=filter2(window,im_max.*imf,'same')-mu_max_muf;

    index1=find(mask1==1);
    index2=find(mask2==1);
    index3=find(mask3==1);

    res1=mu1.*0;
    res2=res1;
    res3=res1;

    res1(index1)=(sigma1f(index1)+C1)./(sqrt(abs(sigma1_sq(index1).*sigmaf_sq(index1)))+C1);
    res2(index2)=(sigma2f(index2)+C2)./(sqrt(abs(sigma2_sq(index2).*sigmaf_sq(index2)))+C2);
    res3(index3)=(sigmaMaxf(index3)+C3)./(sqrt(abs(sigmaMax_sq(index3).*sigmaf_sq(index3)))+C3);

    buffer(:,:,1)=res1;
    buffer(:,:,2)=res2;
    buffer(:,:,3)=res3;

    result=max(buffer,[],3);

    A1=sum(mask1(:));
    A2=sum(mask2(:));
    A3=sum(mask3(:));

    res=sum(result(:))/A3;

    end


    % function [phaseCongruency, or, M, m]=myphasecong3(varargin)
    %
    % This function is a revised version of Kovesi's phasecong3.m.
    % Please "type myphasecong3" for detailed information.
    %
    %
    % Z. Liu @NRCC[ July 31, 2006]

    % PHASECONG2 - Computes edge and corner phase congruency in an image.
    %
    % This function calculates the PC_2 measure of phase congruency.
    % This function supersedes PHASECONG
    %
    % There are potentially many arguments, here is the full usage:
    %
    %   [M m or ft pc EO] = myphasecong3(im, nscale, norient, minWaveLength, ...
    %                         mult, sigmaOnf, dThetaOnSigma, k, cutOff, g)
    %
    % However, apart from the image, all parameters have defaults and the
    % usage can be as simple as:
    %
    %    M = phasecong2(im);
    %
    % Arguments:
    %              Default values      Description
    %
    %    nscale           4    - Number of wavelet scales, try values 3-6
    %    norient          6    - Number of filter orientations.
    %    minWaveLength    3    - Wavelength of smallest scale filter.
    %    mult             2.1  - Scaling factor between successive filters.
    %    sigmaOnf         0.55 - Ratio of the standard deviation of the Gaussian
    %                            describing the log Gabor filter's transfer function
    %                            in the frequency domain to the filter center frequency.
    %    dThetaOnSigma    1.2  - Ratio of angular interval between filter orientations
    %                            and the standard deviation of the angular Gaussian
    %                            function used to construct filters in the
    %                            freq. plane.
    %    k                2.0  - No of standard deviations of the noise energy beyond
    %                            the mean at which we set the noise threshold point.
    %                            You may want to vary this up to a value of 10 or
    %                            20 for noisy images
    %    cutOff           0.5  - The fractional measure of frequency spread
    %                            below which phase congruency values get penalized.
    %    g                10   - Controls the sharpness of the transition in
    %                            the sigmoid function used to weight phase
    %                            congruency for frequency spread.
    %
    % Returned values:
    %    M          - Maximum moment of phase congruency covariance.
    %                 This is used as a indicator of edge strength.
    %    m          - Minimum moment of phase congruency covariance.
    %                 This is used as a indicator of corner strength.
    %    or         - Orientation image in integer degrees 0-180,
    %                 positive anticlockwise.
    %                 0 corresponds to a vertical edge, 90 is horizontal.
    %    ft         - *Not correctly implemented at this stage*
    %                 A complex valued image giving the weighted mean
    %                 phase angle at every point in the image for each
    %                 orientation.
    %    pc         - Cell array of phase congruency images (values between 0 and 1)
    %                 for each orientation
    %    EO         - A 2D cell array of complex valued convolution results
    %
    %   EO{s,o} = convolution result for scale s and orientation o.  The real part
    %   is the result of convolving with the even symmetric filter, the imaginary
    %   part is the result from convolution with the odd symmetric filter.
    %
    %   Hence:
    %       abs(EO{s,o}) returns the magnitude of the convolution over the
    %       image at scale s and orientation o.
    %       angle(EO{s,o}) returns the phase angles.
    %
    % Notes on specifying parameters:
    %
    % The parameters can be specified as a full list eg.
    %  >> [M m or ft pc EO] = phasecong2(im, 5, 6, 3, 2.5, 0.55, 1.2, 2.0, 0.4, 10);
    %
    % or as a partial list with unspecified parameters taking on default values
    %  >> [M m or ft pc EO] = phasecong2(im, 5, 6, 3);
    %
    % or as a partial list of parameters followed by some parameters specified via a
    % keyword-value pair, remaining parameters are set to defaults, for example:
    %  >> [M m or ft pc EO] = phasecong2(im, 5, 6, 3, 'cutOff', 0.3, 'k', 2.5);
    %
    % The convolutions are done via the FFT.  Many of the parameters relate to the
    % specification of the filters in the frequency plane.  The values do not seem
    % to be very critical and the defaults are usually fine.  You may want to
    % experiment with the values of 'nscales' and 'k', the noise compensation factor.
    %
    % Notes on filter settings to obtain even coverage of the spectrum
    % dthetaOnSigma 1.2    norient 6
    % sigmaOnf       .85   mult 1.3
    % sigmaOnf       .75   mult 1.6     (filter bandwidth ~1 octave)
    % sigmaOnf       .65   mult 2.1
    % sigmaOnf       .55   mult 3       (filter bandwidth ~2 octaves)
    %
    % For maximum speed the input image should have dimensions that correspond to
    % powers of 2, but the code will operate on images of arbitrary size.
    %
    % See Also:  PHASECONG, PHASESYM, GABORCONVOLVE, PLOTGABORFILTERS

    % References:
    %
    %     Peter Kovesi, "Image Features From Phase Congruency". Videre: A
    %     Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
    %     Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html
    %
    %     Peter Kovesi, "Phase Congruency Detects Corners and
    %     Edges". Proceedings DICTA 2003, Sydney Dec 10-12

    % April 1996     Original Version written
    % August 1998    Noise compensation corrected.
    % October 1998   Noise compensation corrected.   - Again!!!
    % September 1999 Modified to operate on non-square images of arbitrary size.
    % May 2001       Modified to return feature type image.
    % July 2003      Altered to calculate 'corner' points.
    % October 2003   Speed improvements and refinements.
    % July 2005      Better argument handling, changed order of return values
    % August 2005    Made Octave compatible

    % Copyright (c) 1996-2005 Peter Kovesi
    % School of Computer Science & Software Engineering
    % The University of Western Australia
    % http://www.csse.uwa.edu.au/
    %
    % Permission is hereby  granted, free of charge, to any  person obtaining a copy
    % of this software and associated  documentation files (the "Software"), to deal
    % in the Software without restriction, subject to the following conditions:
    %
    % The above copyright notice and this permission notice shall be included in all
    % copies or substantial portions of the Software.
    %
    % The software is provided "as is", without warranty of any kind.

    %function [phaseCongruency, M, m, or, featType, PC, EO]=myphasecong3(varargin)

    function [phaseCongruency, or, M, m]=myphasecong3(varargin)


    % Get arguments and/or default values
    [im, nscale, norient, minWaveLength, mult, sigmaOnf, ...
                    dThetaOnSigma,k, cutOff, g] = checkargs(varargin(:));

    v = version; Octave = v(1)<'5';  % Crude Octave test
    epsilon         = .0001;         % Used to prevent division by zero.

    thetaSigma = pi/norient/dThetaOnSigma;  % Calculate the standard deviation of the
                                            % angular Gaussian function used to
                                            % construct filters in the freq. plane.

    [rows,cols] = size(im);
    imagefft = fft2(im);              % Fourier transform of image

    zero = zeros(rows,cols);
    totalEnergy = zero;               % Total weighted phase congruency values (energy).
    totalSumAn  = zero;               % Total filter response amplitude values.
    orientation = zero;               % Matrix storing orientation with greatest
                                    % energy for each pixel.
    EO = cell(nscale, norient);       % Array of convolution results.
    covx2 = zero;                     % Matrices for covariance data
    covy2 = zero;
    covxy = zero;

    estMeanE2n = [];
    ifftFilterArray = cell(1,nscale); % Array of inverse FFTs of filters

    % Pre-compute some stuff to speed up filter construction

    % Set up X and Y matrices with ranges normalised to +/- 0.5
    % The following code adjusts things appropriately for odd and even values
    % of rows and columns.
    if mod(cols,2)
        xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
    else
        xrange = [-cols/2:(cols/2-1)]/cols;
    end

    if mod(rows,2)
        yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
    else
        yrange = [-rows/2:(rows/2-1)]/rows;
    end

    [x,y] = meshgrid(xrange, yrange);

    radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
    %radius(rows/2+1, cols/2+1) = 1;   % Get rid of the 0 radius value in the middle
    radius(floor(rows/2)+1,floor(cols/2)+1)=1;  % so that taking the log of the radius will
    % I add the FLOOR here                                % not cause trouble.
    theta = atan2(-y,x);              % Matrix values contain polar angle.
                                    % (note -ve y is used to give +ve
                                    % anti-clockwise angles)
    radius = ifftshift(radius);       % Quadrant shift radius and theta so that filters
    theta  = ifftshift(theta);        % are constructed with 0 frequency at the corners.

    sintheta = sin(theta);
    costheta = cos(theta);
    clear x; clear y; clear theta;    % save a little memory

    % Filters are constructed in terms of two components.
    % 1) The radial component, which controls the frequency band that the filter
    %    responds to
    % 2) The angular component, which controls the orientation that the filter
    %    responds to.
    % The two components are multiplied together to construct the overall filter.

    % Construct the radial filter components...

    % First construct a low-pass filter that is as large as possible, yet falls
    % away to zero at the boundaries.  All log Gabor filters are multiplied by
    % this to ensure no extra frequencies at the 'corners' of the FFT are
    % incorporated as this seems to upset the normalisation process when
    % calculating phase congrunecy.
    lp = lowpassfilter([rows,cols],.45,15);   % Radius .45, 'sharpness' 15

    logGabor = cell(1,nscale);

    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1);
        fo = 1.0/wavelength;                  % Centre frequency of filter.
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));
        logGabor{s} = logGabor{s}.*lp;        % Apply low-pass filter
        logGabor{s}(1,1) = 0;                 % Set the value at the 0 frequency point of the filter
                                            % back to zero (undo the radius fudge).
    end

    % Then construct the angular filter components...

    spread = cell(1,norient);

    for o = 1:norient
    angl = (o-1)*pi/norient;           % Filter angle.

    % For each point in the filter matrix calculate the angular distance from
    % the specified filter orientation.  To overcome the angular wrap-around
    % problem sine difference and cosine difference values are first computed
    % and then the atan2 function is used to determine angular distance.

    ds = sintheta * cos(angl) - costheta * sin(angl);    % Difference in sine.
    dc = costheta * cos(angl) + sintheta * sin(angl);    % Difference in cosine.
    dtheta = abs(atan2(ds,dc));                          % Absolute angular distance.
    spread{o} = exp((-dtheta.^2) / (2 * thetaSigma^2));  % Calculate the
                                                        % angular filter component.
    end

    % The main loop...

    for o = 1:norient                    % For each orientation.
    %fprintf('Processing orientation %d\r',o);
    if Octave fflush(1); end

    angl = (o-1)*pi/norient;           % Filter angle.
    sumE_ThisOrient   = zero;          % Initialize accumulator matrices.
    sumO_ThisOrient   = zero;
    sumAn_ThisOrient  = zero;
    Energy            = zero;

    for s = 1:nscale,                  % For each scale.
        filter = logGabor{s} .* spread{o};   % Multiply radial and angular
                                            % components to get the filter.

    %    if o == 1   % accumulate filter info for noise compensation (nominally the same
                    % for all orientations, hence it is only done once)
            ifftFilt = real(ifft2(filter))*sqrt(rows*cols);  % Note rescaling to match power
            ifftFilterArray{s} = ifftFilt;                   % record ifft2 of filter
    %    end

        % Convolve image with even and odd filters returning the result in EO
        EO{s,o} = ifft2(imagefft .* filter);

        An = abs(EO{s,o});                         % Amplitude of even & odd filter response.
        sumAn_ThisOrient = sumAn_ThisOrient + An;  % Sum of amplitude responses.
        sumE_ThisOrient = sumE_ThisOrient + real(EO{s,o}); % Sum of even filter convolution results.
        sumO_ThisOrient = sumO_ThisOrient + imag(EO{s,o}); % Sum of odd filter convolution results.

        if s==1                                 % Record mean squared filter value at smallest
        EM_n = sum(sum(filter.^2));           % scale. This is used for noise estimation.
        maxAn = An;                           % Record the maximum An over all scales.
        else
        maxAn = max(maxAn, An);
        end

    end                                       % ... and process the next scale

    % Get weighted mean filter response vector, this gives the weighted mean
    % phase angle.

    XEnergy = sqrt(sumE_ThisOrient.^2 + sumO_ThisOrient.^2) + epsilon;
    MeanE = sumE_ThisOrient ./ XEnergy;
    MeanO = sumO_ThisOrient ./ XEnergy;

    % Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
    % using dot and cross products between the weighted mean filter response
    % vector and the individual filter response vectors at each scale.  This
    % quantity is phase congruency multiplied by An, which we call energy.

    for s = 1:nscale,
        E = real(EO{s,o}); O = imag(EO{s,o});    % Extract even and odd
                                                % convolution results.
        Energy = Energy + E.*MeanE + O.*MeanO - abs(E.*MeanO - O.*MeanE);
    end

    % Compensate for noise
    % We estimate the noise power from the energy squared response at the
    % smallest scale.  If the noise is Gaussian the energy squared will have a
    % Chi-squared 2DOF pdf.  We calculate the median energy squared response
    % as this is a robust statistic.  From this we estimate the mean.
    % The estimate of noise power is obtained by dividing the mean squared
    % energy value by the mean squared filter value

    medianE2n = median(reshape(abs(EO{1,o}).^2,1,rows*cols));
    meanE2n = -medianE2n/log(0.5);
    estMeanE2n(o) = meanE2n;

    noisePower = meanE2n/EM_n;                       % Estimate of noise power.

    %  if o == 1
    % Now estimate the total energy^2 due to noise
    % Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))

    EstSumAn2 = zero;
    for s = 1:nscale
        EstSumAn2 = EstSumAn2 + ifftFilterArray{s}.^2;
    end

    EstSumAiAj = zero;
    for si = 1:(nscale-1)
        for sj = (si+1):nscale
        EstSumAiAj = EstSumAiAj + ifftFilterArray{si}.*ifftFilterArray{sj};
        end
    end
    sumEstSumAn2 = sum(sum(EstSumAn2));
    sumEstSumAiAj = sum(sum(EstSumAiAj));

    %  end % if o == 1

    EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2 + 4*noisePower*sumEstSumAiAj;

    tau = sqrt(EstNoiseEnergy2/2);                     % Rayleigh parameter
    EstNoiseEnergy = tau*sqrt(pi/2);                   % Expected value of noise energy
    EstNoiseEnergySigma = sqrt( (2-pi/2)*tau^2 );

    T =  EstNoiseEnergy + k*EstNoiseEnergySigma;       % Noise threshold

    % The estimated noise effect calculated above is only valid for the PC_1 measure.
    % The PC_2 measure does not lend itself readily to the same analysis.  However
    % empirically it seems that the noise effect is overestimated roughly by a factor
    % of 1.7 for the filter parameters used here.

    T = T/1.7;        % Empirical rescaling of the estimated noise effect to
                        % suit the PC_2 phase congruency measure

    Energy = max(Energy - T, zero);          % Apply noise threshold

    % Form weighting that penalizes frequency distributions that are
    % particularly narrow.  Calculate fractional 'width' of the frequencies
    % present by taking the sum of the filter response amplitudes and dividing
    % by the maximum amplitude at each point on the image.

    width = sumAn_ThisOrient ./ (maxAn + epsilon) / nscale;

    % Now calculate the sigmoidal weighting function for this orientation.

    weight = 1.0 ./ (1 + exp( (cutOff - width)*g));


    %----------------------------------------------
    Energy_ThisOrient=weight.*Energy;
    totalSumAn=totalSumAn+sumAn_ThisOrient;
    totalEnergy=totalEnergy+Energy_ThisOrient;

    if (o==1),
        maxEnergy=Energy_ThisOrient;
    else
        change=Energy_ThisOrient>maxEnergy;
        orientation=(o-1).*change+orientation.*(~change);
        maxEnergy=max(maxEnergy, Energy_ThisOrient);
    end
    %----------------------------------------------



    % Apply weighting to energy and then calculate phase congruency

    PC{o} = weight.*Energy./sumAn_ThisOrient;   % Phase congruency for this orientation
    featType{o} = E+i*O;

    % Build up covariance data for every point
    covx = PC{o}*cos(angl);
    covy = PC{o}*sin(angl);
    covx2 = covx2 + covx.^2;
    covy2 = covy2 + covy.^2;
    covxy = covxy + covx.*covy;

    end  % For each orientation

    %fprintf('                                          \r');

    %------------------------------------------------------------
    phaseCongruency=totalEnergy./(totalSumAn+epsilon);
    orientation=orientation*(180/norient);
    %------------------------------------------------------------


    % Edge and Corner calculations

    % The following is optimised code to calculate principal vector
    % of the phase congruency covariance data and to calculate
    % the minimumum and maximum moments - these correspond to
    % the singular values.

    % First normalise covariance values by the number of orientations/2

    covx2 = covx2/(norient/2);
    covy2 = covy2/(norient/2);
    covxy = covxy/norient;   % This gives us 2*covxy/(norient/2)

    denom = sqrt(covxy.^2 + (covx2-covy2).^2)+epsilon;
    sin2theta = covxy./denom;
    cos2theta = (covx2-covy2)./denom;
    or = atan2(sin2theta,cos2theta)/2;    % Orientation perpendicular to edge.
    or = round(or*180/pi);                % Return result rounded to integer
                                        % degrees.
    neg = or < 0;
    or = ~neg.*or + neg.*(or+180);        % Adjust range from -90 to 90
                                        % to 0 to 180.

    M = (covy2+covx2 + denom)/2;          % Maximum moment
    m = (covy2+covx2 - denom)/2;          % ... and minimum moment

    end



    %------------------------------------------------------------------
    % CHECKARGS
    %
    % Function to process the arguments that have been supplied, assign
    % default values as needed and perform basic checks.

    function [im, nscale, norient, minWaveLength, mult, sigmaOnf, ...
            dThetaOnSigma,k, cutOff, g] = checkargs(arg);

        nargs = length(arg);

        if nargs < 1
            error('No image supplied as an argument');
        end

        % Set up default values for all arguments and then overwrite them
        % with with any new values that may be supplied
        im              = [];
        nscale          = 4;     % Number of wavelet scales.
        norient         = 6;     % Number of filter orientations.
        minWaveLength   = 3;     % Wavelength of smallest scale filter.
        mult            = 2.1;   % Scaling factor between successive filters.
        sigmaOnf        = 0.55;  % Ratio of the standard deviation of the
                                % Gaussian describing the log Gabor filter's
                                % transfer function in the frequency domain
                                % to the filter center frequency.
        dThetaOnSigma   = 1.2;   % Ratio of angular interval between filter orientations
                                % and the standard deviation of the angular Gaussian
                                % function used to construct filters in the
                                % freq. plane.
        k               = 2.0;   % No of standard deviations of the noise
                                % energy beyond the mean at which we set the
                                % noise threshold point.
        cutOff          = 0.5;   % The fractional measure of frequency spread
                                % below which phase congruency values get penalized.
        g               = 10;    % Controls the sharpness of the transition in
                                % the sigmoid function used to weight phase
                                % congruency for frequency spread.

        % Allowed argument reading states
        allnumeric   = 1;       % Numeric argument values in predefined order
        keywordvalue = 2;       % Arguments in the form of string keyword
                                % followed by numeric value
        readstate = allnumeric; % Start in the allnumeric state

        if readstate == allnumeric
            for n = 1:nargs
                if isa(arg{n},'char')
                    readstate = keywordvalue;
                    break;
                else
                    if     n == 1, im            = arg{n};
                    elseif n == 2, nscale        = arg{n};
                    elseif n == 3, norient       = arg{n};
                    elseif n == 4, minWaveLength = arg{n};
                    elseif n == 5, mult          = arg{n};
                    elseif n == 6, sigmaOnf      = arg{n};
                    elseif n == 7, dThetaOnSigma = arg{n};
                    elseif n == 8, k             = arg{n};
                    elseif n == 9, cutOff        = arg{n};
                    elseif n == 10,g             = arg{n};
                    end
                end
            end
        end

        % Code to handle parameter name - value pairs
        if readstate == keywordvalue
            while n < nargs

                if ~isa(arg{n},'char') | ~isa(arg{n+1}, 'double')
                    error('There should be a parameter name - value pair');
                end

                if     strncmpi(arg{n},'im'      ,2), im =        arg{n+1};
                elseif strncmpi(arg{n},'nscale'  ,2), nscale =    arg{n+1};
                elseif strncmpi(arg{n},'norient' ,2), norient =   arg{n+1};
                elseif strncmpi(arg{n},'minWaveLength',2), minWavelength = arg{n+1};
                elseif strncmpi(arg{n},'mult'    ,2), mult =      arg{n+1};
                elseif strncmpi(arg{n},'sigmaOnf',2), sigmaOnf =  arg{n+1};
                elseif strncmpi(arg{n},'dthetaOnSigma',2), dThetaOnSigma =  arg{n+1};
                elseif strncmpi(arg{n},'k'       ,1), k =         arg{n+1};
                elseif strncmpi(arg{n},'cutOff'  ,2), cutOff   =  arg{n+1};
                elseif strncmpi(arg{n},'g'       ,1), g        =  arg{n+1};
                else   error('Unrecognised parameter name');
                end

                n = n+2;
                if n == nargs
                    error('Unmatched parameter name - value pair');
                end

            end
        end

        if isempty(im)
            error('No image argument supplied');
        end

        if ~isa(im, 'double')
            im = double(im);
        end

        if nscale < 1
            error('nscale must be an integer >= 1');
        end

        if norient < 1
            error('norient must be an integer >= 1');
        end

        if minWaveLength < 2
            error('It makes little sense to have a wavelength < 2');
        end

        if cutOff < 0 | cutOff > 1
            error('Cut off value must be between 0 and 1');
        end
    end


    %#############################################################################

        function f = lowpassfilter(sze, cutoff, n)

        if cutoff < 0 | cutoff > 0.5
        error('cutoff frequency must be between 0 and 0.5');
        end

        if rem(n,1) ~= 0 | n < 1
        error('n must be an integer >= 1');
        end

        rows = sze(1); cols = sze(2);

        % X and Y matrices with ranges normalised to +/- 0.5
        x =  (ones(rows,1) * [1:cols]  - (fix(cols/2)+1))/cols;
        y =  ([1:rows]' * ones(1,cols) - (fix(rows/2)+1))/rows;

        radius = sqrt(x.^2 + y.^2);        % A matrix with every pixel = radius relative to centre.

        f = fftshift( 1 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   % The filter
        end

    """

    def myphasecong3(
        im,
        nscale=4,
        norient=6,
        minWaveLength=3,
        mult=2.1,
        sigmaOnf=0.55,
        dThetaOnSigma=1.2,
        k=2.0,
        cutOff=0.5,
        g=10,
    ):
        epsilon = 1e-4

        rows, cols = im.shape[-2:]
        imagefft = torch.fft.fft2(im)

        zero = torch.zeros_like(im).to(im.device)
        totalEnergy = zero
        totalSumAn = zero
        orientation = zero

        # Pre-compute some stuff
        x = torch.linspace(-0.5, 0.5, cols)
        y = torch.linspace(-0.5, 0.5, rows)
        x, y = torch.meshgrid(x, y, indexing="ij")

        radius = torch.sqrt(x**2 + y**2).to(im.device)
        theta = torch.atan2(-y, x).to(im.device)

        radius = torch.fft.ifftshift(radius)
        theta = torch.fft.ifftshift(theta)

        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        # Filters
        lp = lowpassfilter((rows, cols), 0.45, 15).to(im.device)
        logGabor = []
        for s in range(nscale):
            wavelength = minWaveLength * mult ** (s)
            fo = 1.0 / wavelength
            logGabor.append(
                (
                    torch.exp(
                        (-((torch.log(radius / fo)) ** 2))
                        / (2 * math.log(sigmaOnf) ** 2)
                    )
                )
            )
            logGabor[s] = logGabor[s] * lp
            logGabor[s][0, 0] = 0

        # Angular filters
        spread = []
        for o in range(norient):
            angl = o * math.pi / norient
            ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
            dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
            dtheta = torch.abs(torch.atan2(ds, dc))
            spread.append(
                torch.exp(
                    (-(dtheta**2)) / (2 * (math.pi / norient / dThetaOnSigma) ** 2)
                )
            )

        # Main loop
        for o in range(norient):
            angl = o * math.pi / norient
            sumE_ThisOrient = zero
            sumO_ThisOrient = zero
            sumAn_ThisOrient = zero
            Energy = zero

            for s in range(nscale):
                filt = (logGabor[s] * spread[o]).transpose(-1, -2)
                EO = torch.fft.ifft2(imagefft * filt)

                An = torch.abs(EO)
                sumAn_ThisOrient += An
                sumE_ThisOrient += EO.real
                sumO_ThisOrient += EO.imag

                if s == 0:
                    maxAn = An
                else:
                    maxAn = torch.maximum(maxAn, An)

            XEnergy = torch.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
            MeanE = sumE_ThisOrient / XEnergy
            MeanO = sumO_ThisOrient / XEnergy

            for s in range(nscale):
                filt = (logGabor[s] * spread[o]).transpose(-1, -2)
                EO = torch.fft.ifft2(imagefft * filt)
                E = EO.real
                O = EO.imag
                Energy += E * MeanE + O * MeanO - torch.abs(E * MeanO - O * MeanE)

            # Apply noise threshold and weighting
            T = 0  # Simplified noise threshold
            Energy = torch.maximum(Energy - T, zero)

            width = sumAn_ThisOrient / (maxAn + epsilon) / nscale
            weight = 1.0 / (1 + torch.exp((cutOff - width) * g))

            Energy_ThisOrient = weight * Energy
            totalSumAn += sumAn_ThisOrient
            totalEnergy += Energy_ThisOrient

            if o == 0:
                maxEnergy = Energy_ThisOrient
            else:
                change = Energy_ThisOrient > maxEnergy
                orientation = (o - 1) * change + orientation * (~change)
                maxEnergy = torch.maximum(maxEnergy, Energy_ThisOrient)

        phaseCongruency = totalEnergy / (totalSumAn + epsilon)
        orientation = orientation * (180 / norient)

        return phaseCongruency, orientation

    def correlation_coefficient(im1, im2, im_max, imf, mask1, mask2, mask3):
        window = torch.ones((11, 11)) / 121  # Simplified Gaussian window

        C1, C2, C3 = 0.0001, 0.0001, 0.0001

        im1 = mask1 * im1
        im2 = mask2 * im2
        im_max = mask3 * im_max

        window = window.to(im1.device)

        mu1 = F.conv2d(
            im1.unsqueeze(0).unsqueeze(0),
            window.unsqueeze(0).unsqueeze(0),
            padding="same",
        ).squeeze()
        mu2 = F.conv2d(
            im2.unsqueeze(0).unsqueeze(0),
            window.unsqueeze(0).unsqueeze(0),
            padding="same",
        ).squeeze()
        muf = F.conv2d(
            imf.unsqueeze(0).unsqueeze(0),
            window.unsqueeze(0).unsqueeze(0),
            padding="same",
        ).squeeze()
        mu_max = F.conv2d(
            im_max.unsqueeze(0).unsqueeze(0),
            window.unsqueeze(0).unsqueeze(0),
            padding="same",
        ).squeeze()

        sigma1_sq = (
            F.conv2d(
                (im1 * im1).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu1**2
        )
        sigma2_sq = (
            F.conv2d(
                (im2 * im2).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu2**2
        )
        sigmaMax_sq = (
            F.conv2d(
                (im_max * im_max).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu_max**2
        )
        sigmaf_sq = (
            F.conv2d(
                (imf * imf).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - muf**2
        )

        sigma1f = (
            F.conv2d(
                (im1 * imf).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu1 * muf
        )
        sigma2f = (
            F.conv2d(
                (im2 * imf).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu2 * muf
        )
        sigmaMaxf = (
            F.conv2d(
                (im_max * imf).unsqueeze(0).unsqueeze(0),
                window.unsqueeze(0).unsqueeze(0),
                padding="same",
            ).squeeze()
            - mu_max * muf
        )

        res1 = torch.zeros_like(mu1)
        res2 = torch.zeros_like(mu1)
        res3 = torch.zeros_like(mu1)

        # res1[mask1] = (sigma1f[mask1] + C1) / (torch.sqrt(torch.abs(sigma1_sq[mask1] * sigmaf_sq[mask1])) + C1)
        # res2[mask2] = (sigma2f[mask2] + C2) / (torch.sqrt(torch.abs(sigma2_sq[mask2] * sigmaf_sq[mask2])) + C2)
        # res3[mask3] = (sigmaMaxf[mask3] + C3) / (torch.sqrt(torch.abs(sigmaMax_sq[mask3] * sigmaf_sq[mask3])) + C3)

        res1 = torch.where(
            mask1,
            (sigma1f + C1) / (torch.sqrt(torch.abs(sigma1_sq * sigmaf_sq)) + C1),
            res1,
        )
        res2 = torch.where(
            mask2,
            (sigma2f + C2) / (torch.sqrt(torch.abs(sigma2_sq * sigmaf_sq)) + C2),
            res2,
        )
        res3 = torch.where(
            mask3,
            (sigmaMaxf + C3) / (torch.sqrt(torch.abs(sigmaMax_sq * sigmaf_sq)) + C3),
            res3,
        )

        result = torch.maximum(res1, torch.maximum(res2, res3))

        return result.sum() / mask3.sum()

    def lowpassfilter(size, cutoff, n):
        rows, cols = size
        x = torch.linspace(-0.5, 0.5, cols)
        y = torch.linspace(-0.5, 0.5, rows)
        x, y = torch.meshgrid(x, y, indexing="ij")

        radius = torch.sqrt(x**2 + y**2)
        return torch.fft.fftshift(1 / (1.0 + (radius / cutoff) ** (2 * n)))

    def _check_type_dim(img):
        # * handled dimensions in conv function
        # _pad_dim = 4 - img.ndim
        # if _pad_dim > 0:
        #     padded_dim = [None for _ in range(_pad_dim)]
        #     img = img[*padded_dim]

        # * type check
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32) / 255.

        return img.type(torch.float32) / 255.0

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    # Main Qp function
    fea_threshold = 0.1

    pc1, _ = myphasecong3(img_A)
    pc2, _ = myphasecong3(img_B)
    pcf, _ = myphasecong3(img_fused)

    mask = pc1 > pc2
    pc_max = torch.where(mask, pc1, pc2)

    mask1 = pc1 > fea_threshold
    mask2 = pc2 > fea_threshold
    mask3 = pc_max > fea_threshold

    resultPC = correlation_coefficient(pc1, pc2, pc_max, pcf, mask1, mask2, mask3)

    return resultPC


def Qcb(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def _check_type_dim(img):
        # * handled dimensions in conv function
        # _pad_dim = 4 - img.ndim
        # if _pad_dim > 0:
        #     padded_dim = [None for _ in range(_pad_dim)]
        #     img = img[*padded_dim]

        # * type check
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32)

        return img.type(torch.float32)

    def gaussian2d(n1, n2, sigma):
        x, y = torch.meshgrid(
            torch.arange(-15, 16).to(im1.device),
            torch.arange(-15, 16).to(im1.device),
            indexing="ij",
        )
        G = torch.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (
            2 * math.pi * sigma * sigma
        )
        return G

    def contrast(G1, G2, im):
        G1 = G1.to(im.device)
        G2 = G2.to(im.device)

        buff = F.conv2d(
            im.unsqueeze(0).unsqueeze(0), G1.unsqueeze(0).unsqueeze(0), padding="same"
        ).squeeze()
        buff1 = F.conv2d(
            im.unsqueeze(0).unsqueeze(0), G2.unsqueeze(0).unsqueeze(0), padding="same"
        ).squeeze()
        return buff / buff1 - 1

    def normalize1(data):
        max = data.max()
        min = data.min()
        # to support vmap
        # if max == min == 0:
        #     return data
        # else:
        newdata = (data - min) / (max - min)
        return torch.round(newdata * 255)

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    im1 = normalize1(img_A)
    im2 = normalize1(img_B)
    fused = normalize1(img_fused)

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622

    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001
    sigma = 2

    hang, lie = im1.shape

    HH = hang / 30
    LL = lie / 30

    u, v = torch.meshgrid(
        torch.linspace(-0.5, 0.5, hang).to(im1.device),
        torch.linspace(-0.5, 0.5, lie).to(im1.device),
        indexing="ij",
    )
    u = LL * u
    v = HH * v
    r = torch.sqrt(u**2 + v**2)

    Sd = torch.exp(-((r / f0) ** 2)) - a * torch.exp(-((r / f1) ** 2))

    fused1 = torch.fft.ifft2(
        torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(im1)) * Sd)
    ).real
    fused2 = torch.fft.ifft2(
        torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(im2)) * Sd)
    ).real
    ffused = torch.fft.ifft2(
        torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(fused)) * Sd)
    ).real

    G1 = gaussian2d(hang, lie, 2)
    G2 = gaussian2d(hang, lie, 4)

    C1 = contrast(G1, G2, fused1).abs()
    C1P = (k * (C1**p)) / (h * (C1**q) + Z)

    C2 = contrast(G1, G2, fused2).abs()
    C2P = (k * (C2**p)) / (h * (C2**q) + Z)

    Cf = contrast(G1, G2, ffused).abs()
    CfP = (k * (Cf**p)) / (h * (Cf**q) + Z)

    mask = (C1P < CfP).float()
    Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

    mask = (C2P < CfP).float()
    Q2F = (C2P / CfP) * mask + (CfP / C2P) * (1 - mask)

    ramda1 = (C1P * C1P) / (C1P * C1P + C2P * C2P)
    ramda2 = (C2P * C2P) / (C1P * C1P + C2P * C2P)

    Q = ramda1 * Q1F + ramda2 * Q2F

    # return Q[torch.bitwise_not(Q.isnan())].mean()
    return safe_mean(Q)


def Qcv(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def normalize1(data):
        max = data.max()
        min = data.min()
        # if max == min == 0:
        #     return data
        # else:
        newdata = (data - min) / (max - min)
        return torch.round(newdata * 255)

    alpha_c = 1
    alpha_s = 0.685
    f_c = 97.3227
    f_s = 12.1653
    windowSize = 16
    alpha = 5

    img_A = img_A.float()
    img_B = img_B.float()
    img_fused = img_fused.float()

    img_A = normalize1(img_A)
    img_B = normalize1(img_B)
    img_fused = normalize1(img_fused)

    flt1 = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(img_A.device)
    )
    flt2 = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(img_A.device)
    )

    fuseX = F.conv2d(img_fused.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
    fuseY = F.conv2d(img_fused.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
    fuseG = torch.sqrt(fuseX**2 + fuseY**2)

    buffer = (fuseX == 0).float() * 0.00001
    fuseX = fuseX + buffer
    fuseA = torch.atan(fuseY / fuseX)

    img1X = F.conv2d(img_A.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
    img1Y = F.conv2d(img_A.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
    im1G = torch.sqrt(img1X**2 + img1Y**2)

    buffer = (img1X == 0).float() * 0.00001
    img1X = img1X + buffer
    im1A = torch.atan(img1Y / img1X)

    img2X = F.conv2d(img_B.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
    img2Y = F.conv2d(img_B.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
    im2G = torch.sqrt(img2X**2 + img2Y**2)

    buffer = (img2X == 0).float() * 0.00001
    img2X = img2X + buffer
    im2A = torch.atan(img2Y / img2X)

    H, W = img_A.shape
    # H = H // windowSize
    # L = W // windowSize

    ramda1 = F.avg_pool2d((im1G**alpha).unsqueeze(0), windowSize).squeeze() * (
        windowSize**2
    )
    ramda2 = F.avg_pool2d((im2G**alpha).unsqueeze(0), windowSize).squeeze() * (
        windowSize**2
    )

    f1 = img_A - img_fused
    f2 = img_B - img_fused

    u, v = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H).to(img_A.device),
        torch.linspace(-0.5, 0.5, W).to(img_A.device),
        indexing="ij",
    )
    u = W / 8 * u
    v = H / 8 * v
    r = torch.sqrt(u**2 + v**2).to(img_A.device)

    theta_m = 2.6 * (0.0192 + 0.144 * r) * torch.exp(-((0.144 * r) ** 1.1))

    index = r == 0
    r[index] = 1

    buff = (0.008 / (r**3) + 1) ** (-0.2)
    buff1 = -0.3 * r * torch.sqrt(1 + 0.06 * torch.exp(0.3 * r))

    theta_d = buff * (1.42 * r * torch.exp(buff1))
    theta_d[index] = 0

    theta_a = alpha_c * torch.exp(-((r / f_c) ** 2)) - alpha_s * torch.exp(
        -((r / f_s) ** 2)
    )

    ff1 = torch.fft.fft2(f1)
    ff2 = torch.fft.fft2(f2)

    Df1 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(ff1) * theta_m)).real
    Df2 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(ff2) * theta_m)).real

    D1 = F.avg_pool2d((Df1**2).unsqueeze(0), windowSize).squeeze()
    D2 = F.avg_pool2d((Df2**2).unsqueeze(0), windowSize).squeeze()

    Q = (ramda1 * D1 + ramda2 * D2).sum() / (ramda1 + ramda2).sum()

    return Q


def Qw(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def _check_type_dim(img):
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32)

        return img.type(torch.float32)

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    def ssim_index(img1, img2, K=None, window=None, L=255):
        if K is None:
            K = [0.01, 0.03]
        if window is None:
            window = gaussian(11, 1.5).to(img1.device)

        C1 = (K[0] * L) ** 2
        C2 = (K[1] * L) ** 2

        mu1 = F.conv2d(img1.unsqueeze(0).unsqueeze(0), window, padding=5)
        mu2 = F.conv2d(img2.unsqueeze(0).unsqueeze(0), window, padding=5)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1.unsqueeze(0).unsqueeze(0).pow(2), window, padding=5) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2.unsqueeze(0).unsqueeze(0).pow(2), window, padding=5) - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1.unsqueeze(0).unsqueeze(0) * img2.unsqueeze(0).unsqueeze(0),
                window,
                padding=5,
            )
            - mu1_mu2
        )

        if C1 > 0 and C2 > 0:
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            )
        else:
            numerator1 = 2 * mu1_mu2 + C1
            numerator2 = 2 * sigma12 + C2
            denominator1 = mu1_sq + mu2_sq + C1
            denominator2 = sigma1_sq + sigma2_sq + C2

            ssim_map = torch.ones_like(mu1)

            idx = denominator1 * denominator2 > 0
            ssim_map[idx] = (numerator1[idx] * numerator2[idx]) / (
                denominator1[idx] * denominator2[idx]
            )

            idx = (denominator1 != 0) & (denominator2 == 0)
            ssim_map[idx] = numerator1[idx] / denominator1[idx]

        mssim = ssim_map.mean()

        return mssim, ssim_map.squeeze(), sigma1_sq.squeeze(), sigma2_sq.squeeze()

    def gaussian(window_size, sigma):
        return _gaussian_kernel_2d(
            1, (window_size, window_size), (sigma, sigma), torch.float32, "cpu"
        )

    ssim, ssim_map, sigma1_sq, sigma2_sq = ssim_index(img_A, img_B)

    buffer = sigma1_sq + sigma2_sq
    test = (buffer == 0).float() * 0.5
    sigma1_sq = sigma1_sq + test
    sigma2_sq = sigma2_sq + test

    buffer = sigma1_sq + sigma2_sq
    ramda = sigma1_sq / buffer

    ssim1, ssim_map1, *_ = ssim_index(img_fused, img_A)
    ssim2, ssim_map2, *_ = ssim_index(img_fused, img_B)

    sw = 3

    if sw == 1:
        Q = ramda * ssim_map1 + (1 - ramda) * ssim_map2
        res = Q.mean()
    elif sw == 2:
        buffer = torch.stack([sigma1_sq, sigma2_sq], dim=-1)
        Cw, _ = torch.max(buffer, dim=-1)
        cw = Cw / Cw.sum()
        Q = (cw * (ramda * ssim_map1 + (1 - ramda) * ssim_map2)).sum()
        res = Q
    elif sw == 3:
        flt1 = (
            torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(img_A.device)
        )
        flt2 = (
            torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(img_A.device)
        )

        fuseX = F.conv2d(img_fused.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
        fuseY = F.conv2d(img_fused.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
        fuseF = torch.sqrt(fuseX**2 + fuseY**2)

        img1X = F.conv2d(img_A.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
        img1Y = F.conv2d(img_A.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
        img1F = torch.sqrt(img1X**2 + img1Y**2)

        img2X = F.conv2d(img_B.unsqueeze(0).unsqueeze(0), flt1, padding=1).squeeze()
        img2Y = F.conv2d(img_B.unsqueeze(0).unsqueeze(0), flt2, padding=1).squeeze()
        img2F = torch.sqrt(img2X**2 + img2Y**2)

        ssim, ssim_map, sigma1_sq, sigma2_sq = ssim_index(img1F, img2F)

        buffer = sigma1_sq + sigma2_sq
        test = (buffer == 0).float() * 0.5
        sigma1_sq = sigma1_sq + test
        sigma2_sq = sigma2_sq + test

        buffer = sigma1_sq + sigma2_sq
        ramda = sigma1_sq / buffer

        ssim1, ssim_map1, *_ = ssim_index(fuseF, img1F)
        ssim2, ssim_map2, *_ = ssim_index(fuseF, img2F)

        buffer = torch.stack([sigma1_sq, sigma2_sq], dim=-1)
        Cw, _ = torch.max(buffer, dim=-1)

        cw = Cw / Cw.sum()
        Qw = (cw * (ramda * ssim_map1 + (1 - ramda) * ssim_map2)).sum()

        alpha = 1
        Qe = Qw**alpha
        res = Qe

    return res


def Qncie(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def _Qncie(img_A, img_B, img_fused):
        img_A = normalize1(img_A)
        img_B = normalize1(img_B)
        img_fused = normalize1(img_fused)

        b = 256
        K = 3

        NCCxy = NCC(img_A, img_B)
        NCCxf = NCC(img_A, img_fused)
        NCCyf = NCC(img_B, img_fused)

        R = torch.tensor([[1, NCCxy, NCCxf], [NCCxy, 1, NCCyf], [NCCxf, NCCyf, 1]])
        r = torch.linalg.eigvals(R).real

        HR = torch.sum(r * torch.log2(r / K) / K)
        HR = -HR / torch.log2(torch.tensor(b))

        NCIE = 1 - HR

        return NCIE

    def NCC(im1, im2):
        im1 = im1.float().cpu()
        im2 = im2.float().cpu()

        N = 256
        b = 256

        h = torch.histogramdd(
            torch.stack([im1.flatten(), im2.flatten()], dim=1),
            bins=N,
            range=(0, 255, 0, 255),
        )[0]  # batching is not supported for vmap
        h = h / h.sum()

        im1_marg = h.sum(dim=1)
        im2_marg = h.sum(dim=0)

        H_x = -torch.sum(im1_marg * torch.log2(im1_marg + (im1_marg == 0)))
        H_y = -torch.sum(im2_marg * torch.log2(im2_marg + (im2_marg == 0)))

        H_xy = -torch.sum(h * torch.log2(h + (h == 0)))
        H_xy = H_xy / torch.log2(torch.tensor(b))

        H_x = H_x / torch.log2(torch.tensor(b))
        H_y = H_y / torch.log2(torch.tensor(b))

        return H_x + H_y - H_xy

    def normalize1(data):
        data = data.float()
        max = data.max()
        min = data.min()
        # if max == min == 0:
        #     return data
        # else:
        newdata = (data - min) / (max - min)
        return torch.round(newdata * 255)

    res = _Qncie(img_A, img_B, img_fused)

    return res


def Qy(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def _check_type_dim(img):
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32)
        return img.type(torch.float32)

    def ssim_index(img1, img2, K=None, window=None, L=255):
        if K is None:
            K = [0.01, 0.03]
        if window is None:
            window = gaussian(11, 1.5).to(img1.device)

        C1 = (K[0] * L) ** 2
        C2 = (K[1] * L) ** 2

        mu1 = F.conv2d(img1.unsqueeze(0).unsqueeze(0), window, padding=5)
        mu2 = F.conv2d(img2.unsqueeze(0).unsqueeze(0), window, padding=5)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1.unsqueeze(0).unsqueeze(0).pow(2), window, padding=5) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2.unsqueeze(0).unsqueeze(0).pow(2), window, padding=5) - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1.unsqueeze(0).unsqueeze(0) * img2.unsqueeze(0).unsqueeze(0),
                window,
                padding=5,
            )
            - mu1_mu2
        )

        if C1 > 0 and C2 > 0:
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            )
        else:
            numerator1 = 2 * mu1_mu2 + C1
            numerator2 = 2 * sigma12 + C2
            denominator1 = mu1_sq + mu2_sq + C1
            denominator2 = sigma1_sq + sigma2_sq + C2

            ssim_map = torch.ones_like(mu1)

            idx = denominator1 * denominator2 > 0
            ssim_map[idx] = (numerator1[idx] * numerator2[idx]) / (
                denominator1[idx] * denominator2[idx]
            )

            idx = (denominator1 != 0) & (denominator2 == 0)
            ssim_map[idx] = numerator1[idx] / denominator1[idx]

        mssim = ssim_map.mean()

        return mssim, ssim_map.squeeze(), sigma1_sq.squeeze(), sigma2_sq.squeeze()

    def gaussian(window_size, sigma):
        return _gaussian_kernel_2d(
            1, (window_size, window_size), (sigma, sigma), torch.float32, "cpu"
        )

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    mssim1, ssim_map1, sigma1_sq1, sigma2_sq1 = ssim_index(img_A, img_B)
    mssim2, ssim_map2, sigma1_sq2, sigma2_sq2 = ssim_index(img_A, img_fused)
    mssim3, ssim_map3, sigma1_sq3, sigma2_sq3 = ssim_index(img_B, img_fused)

    bin_map = (ssim_map1 >= 0.75).float()
    ramda = sigma1_sq1 / (sigma1_sq1 + sigma2_sq1)

    Q1 = (ramda * ssim_map2 + (1 - ramda) * ssim_map3) * bin_map
    Q2 = torch.max(ssim_map2, ssim_map3) * (1 - bin_map)

    A = Q1 + Q2
    # A = A[~torch.isnan(A)]
    # A = A[~torch.isinf(A)]
    # Q = A.mean()
    Q = safe_mean(A)

    return Q


def NMI(img_A: torch.Tensor, img_B: torch.Tensor, img_fused: torch.Tensor):
    def _check_type_dim(img):
        # if img.dtype != torch.float32 or img.max() > 1 or img.dtype == torch.uint8:
        #     img = img.type(torch.float32)
        return img.type(torch.float32)

    def mutual_info(im1, im2):
        N = 256

        im1 = im1.float().cpu()
        im2 = im2.float().cpu()

        h = torch.histogramdd(
            torch.stack([im1.flatten(), im2.flatten()], dim=1),
            bins=N,
            range=[0, 255, 0, 255],
        )[0]
        h = h / h.sum()

        im1_marg = h.sum(dim=1)
        im2_marg = h.sum(dim=0)

        H_x = -torch.sum(im1_marg * torch.log2(im1_marg + (im1_marg == 0)))
        H_y = -torch.sum(im2_marg * torch.log2(im2_marg + (im2_marg == 0)))

        H_xy = -torch.sum(h * torch.log2(h + (h == 0)))

        MI = H_x + H_y - H_xy

        return MI, H_xy, H_x, H_y

    img_A, img_B, img_fused = map(_check_type_dim, [img_A, img_B, img_fused])

    I_fx, H_xf, H_x, H_f1 = mutual_info(img_A, img_fused)
    I_fy, H_yf, H_y, H_f2 = mutual_info(img_B, img_fused)

    MI = 2 * (I_fx / (H_f1 + H_x) + I_fy / (H_f2 + H_y))
    return MI


def FMI(ima, imb, imf, feature="edge", w=3):
    ima = ima.float()
    imb = imb.float()
    imf = imf.float()

    def feature_extraction(img, feature):
        if feature == "none":
            return img
        elif feature == "gradient":
            return gradient(img)
        elif feature == "edge":
            return edge(img)
        # Add more feature extraction methods as needed

    def gradient(img):
        kernel = (
            torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(img.device)
        )
        return torch.abs(
            F.conv2d(img.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()
        )

    def edge(img):
        kernel = (
            torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(img.device)
        )
        return torch.abs(
            F.conv2d(img.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()
        )

    def calculate_fmi(sub1, sub2):
        # Calculate PDFs
        pdf1 = sub1.flatten() / sub1.sum()
        pdf2 = sub2.flatten() / sub2.sum()

        # Calculate CDFs
        cdf1 = torch.cumsum(pdf1, 0)
        cdf2 = torch.cumsum(pdf2, 0)

        # Calculate correlation
        c = pearson_correlation(pdf1, pdf2)

        # Calculate joint entropy and marginal entropies
        joint_entropy = calculate_joint_entropy(cdf1, cdf2, c)
        entropy1 = calculate_entropy(pdf1)
        entropy2 = calculate_entropy(pdf2)

        # Calculate mutual information
        mi = entropy1 + entropy2 - joint_entropy

        # Calculate normalized mutual information
        return 2 * mi / (entropy1 + entropy2)

    def normalize(sub):
        sub_min = sub.min()
        sub_max = sub.max()
        # if sub_max == sub_min:
        #     return torch.ones_like(sub)
        return (sub - sub_min) / (sub_max - sub_min)

    def pearson_correlation(x, y):
        return F.cosine_similarity(x - x.mean(), y - y.mean(), dim=0)

    def calculate_joint_entropy(cdf1, cdf2, c):
        return -torch.sum(cdf1 * cdf2 * torch.log2(cdf1 * cdf2 + 1e-10))

    def calculate_entropy(pdf):
        return -torch.sum(pdf * torch.log2(pdf + 1e-10))

    # Feature Extraction
    aFeature = feature_extraction(ima, feature)
    bFeature = feature_extraction(imb, feature)
    fFeature = feature_extraction(imf, feature)

    # Sliding window
    m, n = aFeature.shape
    w = w // 2
    fmi_map = torch.ones(m - 2 * w, n - 2 * w)

    # unfold the feature maps
    patch_size = 2 * w + 1
    stride = 1

    # extract the patches from the feature maps
    a_patches = F.unfold(
        aFeature.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=stride
    )
    b_patches = F.unfold(
        bFeature.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=stride
    )
    f_patches = F.unfold(
        fFeature.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=stride
    )

    # reshape the patches to the original size
    a_patches = a_patches.view(-1, patch_size, patch_size)
    b_patches = b_patches.view(-1, patch_size, patch_size)
    f_patches = f_patches.view(-1, patch_size, patch_size)

    # normalize the patches
    a_patches = normalize(a_patches)
    b_patches = normalize(b_patches)
    f_patches = normalize(f_patches)

    # using vmap to parallelize the calculation of FMI
    from torch.func import vmap

    fmi_af = vmap(calculate_fmi)(a_patches, f_patches)
    fmi_bf = vmap(calculate_fmi)(b_patches, f_patches)

    # calculate the average FMI
    fmi_map = (fmi_af + fmi_bf) / 2

    # reshape fmi_map to the original size
    fmi_map = fmi_map.view(1, 1, m - 2 * w, n - 2 * w).squeeze()

    # fmi_map = fmi_map[~torch.isnan(fmi_map)]
    # fmi_map = fmi_map[~torch.isinf(fmi_map)]
    # return fmi_map.mean()

    return safe_mean(fmi_map)


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = (
        F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
        - muY_sq_seq
    )
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = (
        F.conv2d(X * X, window, padding=ws // 2, groups=C).view(C, H, W) - muX_sq
    )

    # compute correlation term
    sigmaXY = (
        F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(
            K, C, H, W
        )
        - muX.expand_as(muY_seq) * muY_seq
    )

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(C, H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(-((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q


from math import exp


def gaussian(window_size, sigma):
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class MEFSSIM(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        channel=3,
        sigma_g=0.2,
        sigma_l=0.2,
        c1=0.01,
        c2=0.03,
        is_lum=False,
    ):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(
            X,
            Ys,
            window,
            self.window_size,
            self.denom_g,
            self.denom_l,
            self.C1,
            self.C2,
            self.is_lum,
        )


def mef_ssim(img_A, img_B, img_fused):
    mefssim = MEFSSIM().cuda()

    def dim_type_check(img):
        if img.ndim == 2:
            img = img[None, None]

        if img.dtype != torch.float32:
            img = img.float() / 255.0
        elif (
            img.dtype == torch.float32 and img.max() > 1.2
        ):  # assume 1.2 is the max value
            img = img / 255.0

        return img

    img_A = dim_type_check(img_A)
    img_B = dim_type_check(img_B)
    img_fused = dim_type_check(img_fused)

    return mefssim(img_fused, img_A) + mefssim(img_fused, img_B)


def to_batched(img):
    ndim = img.ndim
    if ndim == 2:
        return img[None, None].repeat(1, 3, 1, 1)
    elif ndim == 3:
        return img[None]
    elif ndim == 4:
        return img
    else:
        raise ValueError(f"Invalid number of dimensions for batched image: {ndim}")


@torch.no_grad()
def LPIPS(
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    img_fused: torch.Tensor,
    force_to_reinit=False,
):
    if "lpips" not in _metric_may_init_lst or force_to_reinit:
        init_lpips()
    lpips = _metric_may_init_lst["lpips"]

    def _update(img1, img2):
        return lpips(img1, img2)

    # to [0, 1]
    img_A = img_A / 255.0
    img_B = img_B / 255.0
    img_fused = img_fused / 255.0

    # to batched
    img_A = to_batched(img_A)
    img_B = to_batched(img_B)
    img_fused = to_batched(img_fused)

    return _update(img_A, img_fused) + _update(img_B, img_fused)


@torch.no_grad()
def FID(
    fid: FrechetInceptionDistance,
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    img_fused: torch.Tensor,
) -> callable:
    # img value range is [0, 255]
    def get_result():
        # call .compute() to get the result
        return fid.compute()

    def _update(fuse_img, real_img):
        fid.update(fuse_img, real=False)
        fid.update(real_img, real=True)

    img_A = to_batched(img_A).to(torch.uint8)
    img_B = to_batched(img_B).to(torch.uint8)
    img_fused = to_batched(img_fused).to(torch.uint8)

    _update(img_fused, img_A)
    _update(img_fused, img_B)

    return get_result


@torch.no_grad()
def CLIPIQA(img_fused: torch.Tensor, backend: str = "pyiqa"):
    if "clipiqa" not in _metric_may_init_lst:
        init_clipiqa(device=img_fused.device, backend=backend)
    clipiqa = _metric_may_init_lst["clipiqa"]

    if backend == "torchmetrics":
        return clipiqa(img_fused)
    else:
        return clipiqa(to_batched(img_fused) / 255.0)


@torch.no_grad()
def BRISQUE(img_fused: torch.Tensor, backend: str = "pyiqa"):
    if "brisque" not in _metric_may_init_lst:
        init_brisque(backend, device=img_fused.device)
    brisque = _metric_may_init_lst["brisque"]
    device = img_fused.device
    if backend == "self":
        assert img_fused.dim() in (2, 3), "img_fused should be 2D or 3D"
        img_fused = img_fused.cpu().numpy().astype(np.uint8)
        return torch.tensor(brisque.get_score(img_fused)).to(device)
    else:  # use iqa, [b, 3, h, w], 0 - 1
        return brisque(to_batched(img_fused).float() / 255.0)


@torch.no_grad()
def MUSIQ(img_fused: torch.Tensor):
    if "musiq" not in _metric_may_init_lst:
        init_musiq(device=img_fused.device)

    musiq = _metric_may_init_lst["musiq"]
    return musiq(to_batched(img_fused).float() / 255.0)


@torch.no_grad()
def NIQE(img_fused: torch.Tensor):
    if "niqe" not in _metric_may_init_lst:
        init_niqe(device=img_fused.device)

    niqe = _metric_may_init_lst["niqe"]
    return niqe(to_batched(img_fused).float() / 255.0)


# ============================== perceptual metric initialization ==============================
_metric_may_init_lst = {}


def init_lpips():
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    _metric_may_init_lst["lpips"] = lpips
    print("LPIPS initialized")
    return lpips


def init_fid():
    fid = FrechetInceptionDistance(normalize=False).cuda()
    _metric_may_init_lst["fid"] = fid
    print("FID initialized")

    return fid


def init_clipiqa(
    prompt: str | tuple[str] = "quality", backend: str = "clipiqa", device="cuda"
):
    if isinstance(prompt, str):
        prompt = (prompt,)
    if backend == "torchmetrics":
        clipiqa = CLIPImageQualityAssessment(
            model_name_or_path="clip_iqa", prompts=prompt
        ).to(device)
    elif backend == "pyiqa":
        clipiqa = create_metric("clipiqa", device=device)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    _metric_may_init_lst["clipiqa"] = clipiqa
    print("CLIPIQA initialized")

    return clipiqa


def init_brisque(backend: str = "pyiqa", device="cuda"):
    use_iqa = backend == "pyiqa"
    if not use_iqa:
        from utils.fusion_index.brisque import BRISQUE as BRISQUE_FN

        brisque = BRISQUE_FN()
    else:
        brisque = create_metric("brisque", device=device)

    _metric_may_init_lst["brisque"] = brisque
    print("BRISQUE initialized")

    return _metric_may_init_lst["brisque"]


def init_musiq(device="cuda"):
    _metric_may_init_lst["musiq"] = create_metric("musiq", device=device)
    print("MUSIQ initialized")

    return _metric_may_init_lst["musiq"]


def init_niqe(device="cuda"):
    _metric_may_init_lst["niqe"] = create_metric("niqe", device=device)
    print("NIQE initialized")

    return _metric_may_init_lst["niqe"]


######################################## function entry ########################################


def evaluate_MEF_MFF_metric_torch(
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    img_fused: torch.Tensor,
    metrics: "list[str] | str" = "all",
    reinit_cls_metrics: bool = False,
):
    if "all" == metrics:
        metrics = [
            "Qc",
            "Qp",
            "Qcb",
            "Qcv",
            "Qw",
            "Qncie",
            "Qy",
            "NMI",
            "FMI",
            "LPIPS",
            "FID",
            "CLIPIQA",
        ]

    results = {}
    # * ==========================================================
    # * Reference fusion metrics

    # if "CE" in metrics:
    #     results["CE"] = CrossEN(img_A, img_B, img_fused).item()
    if "Qc" in metrics:
        results["Qc"] = Qc(img_A, img_B, img_fused)  # .item()
    if "Qp" in metrics:
        results["Qp"] = Qp(img_A, img_B, img_fused)  # .item()
    if "Qcb" in metrics:
        results["Qcb"] = Qcb(img_A, img_B, img_fused)  # .item()
    if "Qcv" in metrics:
        results["Qcv"] = Qcv(img_A, img_B, img_fused)  # .item()
    if "Qw" in metrics:
        results["Qw"] = Qw(img_A, img_B, img_fused)  # .item()
    if "Qncie" in metrics:
        results["Qncie"] = Qncie(img_A, img_B, img_fused)  # .item()
    if "Qy" in metrics:
        results["Qy"] = Qy(img_A, img_B, img_fused)  # .item()
    if "NMI" in metrics:
        results["NMI"] = NMI(img_A, img_B, img_fused)  # .item()
    if "FMI" in metrics:
        results["FMI"] = FMI(img_A, img_B, img_fused)  # .item()
    if "MEFSSIM" in metrics:
        results["MEFSSIM"] = mef_ssim(img_A, img_B, img_fused)
    # * ==========================================================
    # * Perception metrics

    if "LPIPS" in metrics:
        results["LPIPS"] = LPIPS(img_A, img_B, img_fused)  # .item()
    if "FID" in metrics:
        # ready to call the returned function to get the result
        if reinit_cls_metrics:
            fid = init_fid()
        else:
            assert "fid" in _metric_may_init_lst, "FID is not initialized"
            fid = _metric_may_init_lst["fid"]
        results["FID"] = FID(fid, img_A, img_B, img_fused)
    # * ==========================================================
    # * Non-reference IQA metrics

    if "CLIPIQA" in metrics:
        results["CLIPIQA"] = CLIPIQA(img_fused)
    if "BRISQUE" in metrics:
        results["BRISQUE"] = BRISQUE(img_fused, backend="pyiqa")
    if "MUSIQ" in metrics:
        results["MUSIQ"] = MUSIQ(img_fused)
    if "NIQE" in metrics:
        results["NIQE"] = NIQE(img_fused)

    return results


if __name__ == "__main__":
    torch.manual_seed(2024)
    torch.cuda.set_device(1)

    # img_A = torch.randint(0, 256, (256, 384)).cuda()
    # noise = torch.randint(-1, 1, (256, 384)).cuda()
    # img_B = (img_A + noise).clamp(0, 255)
    # img_fused = (img_A + img_B).clamp(0, 255)

    from torchvision.io import read_image

    img_fusion = (
        read_image(
            "/Data4/cao/ZiHanCao/exps/panformer/task_datasets/DIF/data/hier_test/deg_heavy_1_orig_size/RealMFF/gaussian noise/gt/00061_212.jpg"
        )
        .float()
        .cuda()
    )

    # print('CE:', CrossEN(img_A, img_B, img_fused))
    # print('Qc:', Qc(img_A, img_B, img_fused))
    # print('Qp:', Qp(img_A, img_B, img_fused))
    # print('Qcb:', Qcb(img_A, img_B, img_fused))
    # print('Qcv:', Qcv(img_A, img_B, img_fused))
    # print('Qw:', Qw(img_A, img_B, img_fused))
    # print('Qncie:', Qncie(img_A, img_B, img_fused))
    # print('Qy:', Qy(img_A, img_B, img_fused))
    # print('NMI:', NMI(img_A, img_B, img_fused))
    # print('FMI:', FMI(img_A, img_B, img_fused))
    # print('LPIPS:', LPIPS(img_A, img_B, img_fused))
    # print('FID:', FID(img_A, img_B, img_fused)())

    # print(CLIPIQA(img_A, img_B, img_fused))
    # print(mef_ssim(img_A, img_B, img_fused))
    # print(BRISQUE(img_fused, backend='pyiqa'))
    # print(CLIPIQA(img_fusion))
    print(BRISQUE(img_fusion, backend="pyiqa"))
