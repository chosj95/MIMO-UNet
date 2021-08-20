# MIMO-UNet - Official Pytorch Implementation

<img src= "https://github.com/chosj95/MIMO-UNet/blob/main/img/Architecture.jpg" width="80%">

This repository provides the official PyTorch implementation of the following paper:

> Rethinking Coarse-to-Fine Approach in Single Image Deblurring
>
> [Sung-Jin Cho](https://github.com/chosj95) *, [Seo-Won Ji](https://scholar.google.co.kr/citations?user=3n3Zpl8AAAAJ&hl) *, [Jun-Pyo Hong](https://scholar.google.com/citations?hl=ko&user=trMOhfsAAAAJ), [Seung-Won Jung](https://scholar.google.co.kr/citations?user=2PHpYPQAAAAJ&hl), [Sung-Jea Ko](https://scholar.google.co.kr/citations?user=oLRJMVMAAAAJ&hl)
> 
> In ICCV 2021. (* indicates equal contribution)
>
> Paper: https://arxiv.org/abs/2108.05054
>
> Abstract: Coarse-to-fine strategies have been extensively used for the architecture design of single image deblurring networks. Conventional methods typically stack sub-networks with multi-scale input images and gradually improve sharpness of images from the bottom sub-network to the top sub-network, yielding inevitably high computational costs. Toward a fast and accurate deblurring network design, we revisit the coarse-to-fine strategy and present a multi-input multi-output U-net (MIMO-UNet). The MIMO-UNet has three distinct features. First, the single encoder of the MIMO-UNet takes multi-scale input images to ease the difficulty of training. Second, the single decoder of the MIMO-UNet outputs multiple deblurred images with different scales to mimic multi-cascaded U-nets using a single U-shaped network. Last, asymmetric feature fusion is introduced to merge multi-scale features in an efficient manner. Extensive experiments on the GoPro and RealBlur datasets demonstrate that the proposed network outperforms the state-of-the-art methods in terms of both accuracy and computational complexity.

---
## Contents
The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Dataset](#Dataset)
3. [Train](#Train)
4. [Test](#Test)
5. [Performance](#Performance)
5. [Model](#Model)


---
## Dependencies

- Python
- Pytorch (1.4)
  - Different versions may cause some errors.
- scikit-image
- opencv-python
- Tensorboard

---
## Dataset

- Download deblur dataset from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) .
- Unzip files ```dataset``` folder.
- Preprocess dataset by running the command below:
  
  ``` python data/preprocessing.py```

After preparing data set, the data folder should be like the format below:
```
GOPRO
├─ train
│ ├─ blur    % 2103 image pairs
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ sharp
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ test    % 1111 image pairs
│ ├─ ...... (same as train)

```

---
## Train

To train MIMO-UNet+ , run the command below:

``` python main.py --model_name "MIMO-UNetPlus" --mode "train" --data_dir "dataset/GOPRO" ```

or to train MIMO-UNet, run the command below:

``` python main.py --model_name "MIMO-UNet" --mode "train" --data_dir "dataset/GOPRO" ```

Model weights will be saved in ``` results/model_name/weights``` folder.

---

## Test

To test MIMO-UNet+ , run the command below:

``` python main.py --model_name "MIMO-UNetPlus" --mode "test" --data_dir "dataset/GOPRO" --test_model "MIMO-UNetPlus.pkl" ```

or to test MIMO-UNet, run the command below:

``` python main.py --model_name "MIMO-UNet" --mode "test" --data_dir "dataset/GOPRO" --test_model "MIMO-UNet.pkl" ```

Output images will be saved in ``` results/model_name/result_image``` folder.

---

## Performance

<img src= "https://github.com/chosj95/MIMO-UNet/blob/main/img/Graph.jpg" width="50%">

|     Method      | MIMO-UNet     | MIMO-UNet+ | MIMO-UNet++ | 
| :---------------: | :-----: |  :-----------------------------: | :----------------------: | 
|     PSNR (dB)     | 31.73          | 32.45                       | 32.68                 | 
|   SSIM  | 0.951  |  0.957                    | 0.959    | 
| Runtime (s) | 0.008   |  0.017                        | 0.040                | 


---

## Model

We provide our pre-trained models. 
You can test our network following the instruction above.

- MIMO-UNet: https://drive.google.com/file/d/1EQJoQj3YMLFfzrbgzWMD3Xj96RqLdIlx/view?usp=sharing
- MIMO-UNet+: https://drive.google.com/file/d/166sufeHcdDTgXHNbCRzTC4T6DzuflB5m/view?usp=sharing



