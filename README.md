# Unify Language and Mask Guidance in an Efficient Network [TPAMI 2025]

<div align="center">
  <a href=https://scholar.google.com/citations?user=pv61p_EAAAAJ&hl=en> Zihan Cao </a> |
  <a href=https://scholar.google.com/citations?user=E5KO9XsAAAAJ&hl=en> Yu-Jie Liang </a> |
  <a href=https://scholar.google.com/citations?user=TZs9NxkAAAAJ&hl=en> Liang-Jian Deng </a> |
  <a href=https://scholar.google.com/citations?user=sjb_uAMAAAAJ&hl=en> Gemine Vivone</>

  <a>University of Electronic Science and Technology of China (UESTC) </a><br>
  <a>Institute of Methodologies for Environmental Analysis, CNR-IMAA </a>
</div>

[[Paper](assets/paper.pdf)] [[Data](https://huggingface.co/datasets/iamzihan/RWKVFusionDataset/tree/main)] [[Benchmark](https://pan.baidu.com/s/1Tfbc1xTu2njYrTxwVWdkvQ?pwd=djij)]

Official implementation of [Unify language and mask guidance in an efficient network (TPAMI)](https://ieeexplore.ieee.org/document/11091495).

<!-- Images -->
<div align="center">
<img src="assets/radar_results.png" width="800">
<p><em>Figure 1: Results comparisons.</em></p>
</div>

<div align="center">
<img src="assets/architecture.png" width="800">
<p><em>Figure 2: RWKVFusion network architecture.</em></p>
</div>


# News
**[2025/09/28]**: The benckmark results are released.

**[2025/09/25]**: The data preparation code is released.

**[2025/09/11]**: WIP - We released the training and inference code of RWKVFusion. We are now preparing the pre-trained checkpoints and used dataset. We will soon release the results. Please stay tuned.


# Fast Run


## Data preparation

The raw datasets can be access:
- Pansharpening: we follow the [[Pan-Collection]](https://github.com/liangjiandeng/PanCollection), please download the H5 data format dataset. Datasets used at paper are: WV3, GF2, and QB.
- HMIF: we follow the [[PSRT]](https://pan.baidu.com/s/1SLR0QKVyMWOOuYIYgcNv_Q?pwd=7ja6#list/path=%2F) paper, please download the Chikusei x4 and Pavia x4 datasets.
- VIF, MEF, MFF, and MIF tasks, thanks for previous works, we collected VIF-LLVIP, VIF-M3FD, VIF-MSRS, VIF-RoadScene, VIF-TNO, MEF-SICE, MEF-MEFB, MFF-RealMFF, MFF-Lytro, MFF-MFFW, and MIF-Harvard. You can download them from [[Huggingface]](https://huggingface.co/datasets/iamzihan/RWKVFusionDataset/tree/main).

For mask and caption preparation for the RWKVFusion model,
please refer the [data preparation](data_preparation/README.md) section.

For your convience, we provide the preprocessed data:
you can download at [[Huggingface]](https://huggingface.co/datasets/iamzihan/RWKVFusionDataset/tree/main)

The overall data structure should be:
<details>
  <summary>Data structure (click to unfold)</summary>
<pre><code>
data
├── HMIF
│   ├── Cheikusei
│   └── Pavia
├── MEF
│   ├── MEF-MEFB
│   └── MEF-SICE
├── MFF
│   ├── MFF-Lytro
│   ├── MFF-MFFW
│   ├── MFF-RealMFF
│   └── MFI-WHU
├── MIF
│   └── MedHarvard
├── pansharpening
│   ├── gf
│   ├── qb
│   └── wv3
└── VIF
    ├── VIF-LLVIP
    ├── VIF-MSRS
    ├── VIF-RoadScene_and_TNO
    └── VIF-TNO
</code></pre>
</details>

## Training
You can simply run different tasks (VIF, MEF, MFF, Medical image fusion, Pansharpening, and HISR) with the following command:
```bash
sh scripts/train/train_rwkvfusion_VIF.sh
```
to start training on VIF task.
Please check the [`scripts/train`](scripts/train) directory for more details.


## Inference
After we release the trained models, you can simply run different commands to inference RWKVFusion models:
```bash
sh scripts/test/sharpening.sh
```
to start inference on Pansharpening task, for example. Please check the [`scripts/test`](scripts/test) directory for more details.


## Metrics
We rewrite the VIF, MEF, and MFF metrics in Python, which originally implemented in MATLAB. Please check the [`scripts/py_scripts`](scripts/py_scripts) directory for more details.

For Pansharpening and HISR tasks, we use the [Pan-collection](https://github.com/liangjiandeng/PanCollection) provided [Matlab package](matlab_test_pkgs/Pansharpening_Hyper_SR_Matlab_Test_Package) for evaluation.

# Results
You can access the results of all tasks of RWKVFusion at [BaiduYunDisk](https://pan.baidu.com/s/1Tfbc1xTu2njYrTxwVWdkvQ?pwd=djij) (Extract code: djij).

# Citations
If you find this work useful, please kindly cite our paper:
```bibtex
@ARTICLE{RWKVFusion,
  author={Cao, Zi-Han and Liang, Yu-Jie and Deng, Liang-Jian and Vivone, Gemine},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={An Efficient Image Fusion Network Exploiting Unifying Language and Mask Guidance},
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3591930}
}
```
