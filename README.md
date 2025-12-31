## <div align="center">DRNet: Dynamic Routing for Robust Multispectral Object Detection under Modality Missing</div>

### Introduction
We propose a Dynamic Routing-based Multispectral Object Detection Network (DRNet), which is evaluated on the FLIR, KAIST, and LLVIP datasets. To address the issue of modality missing in RGB-T networks, DRNet introduces modality awareness and dynamic adaptation mechanisms. The network incorporates a Modality-Aware Dynamic Routing (MADR) module, which adaptively routes features between multimodal fusion and unimodal enhancement pathways to reduce the impact of missing modalities. Furthermore, we propose a Unified Feature Harmonize (UFH) module that equips a shared detection head with modality-aware transformations, allowing it to adapt to fused, RGB-only, and thermal-only inputs by capturing modality-specific feature distributions, thereby improving cross-modal robustness. In our implementation, Dynamic Feature Ehancement (DFE)  module and Dynamic Feature Mapping (DFM) module correspond to the concepts MADR and UFH introduced in the paper.

### Overview
<div align="center">
  <img src="./diagram.jpg"  widt=="600" height="400" >
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of DRNet framework. </div>
</div>

<div align="center">
  <img src="./FE.jpg"   widt=="600" height="400" >
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 2. Illustration of the feature enhancement modules of MADR module corresponding to FE1, FE2, FE3 in Fig 1. </div>
</div>

### Installation
We use "conda create --name DRNet python=3.8" to create a environment and install the dependencies in requirements.txt. 

### Datasets
The datasets encompass samples for both training and testing, covering scenarios with complete modalities and those with missing modalities.
We define the modality-missing rates on the three multimodal datasets as 0, 5%, 10%, 15%, 25%, 35%, 40%, 45%, 50%, and 100%. This spectrum of rates comprises the full-modality scenario (0%), scenarios with partial modality missing, and the scenarios of complete RGB or thermal missing (100%). For each sample, only one modality is randomly dropped at a time, ensuring the two modalities never fail simultaneously. The missing modality is replaced with a zero-filled tensor (i.e., all pixel values are set to zero). The modality-missing data are then mixed with intact multimodal data to form new training and test sets containing partial modality missing. We denote the overall missing condition by the pair T:V, where T and V represent the missing rates for the thermal and RGB modalities, respectively. This results in 11 dataset variants: 0:0, 5:5, 10:10, 15:15, 25:25, 35:35, 40:40, 45:45, 50:50, 100:0, and 0:100. Additionally, based on the test sets of the three original datasets, we generated two specific failure scenarios: complete RGB missing in daytime scenes and complete thermal missing in nighttime scenes.

- **KAIST**  Link：https://pan.baidu.com/s/1DY7YkLm0yvO-0osv04m9zg Code：axui 
- **FLIR**    Link：https://pan.baidu.com/s/1xkdnuPpRwSabuxYmQaT9Ig Code: kiah 
- **LLVIP**  Link: https://pan.baidu.com/s/1JzzfX-S5X0zQcP7KWWfyhQ Code: emjy

Download the datasets to your disk, the organized directory should look like:
```
    --datasetname:
    	|--infrared
         |--train
         |--test
      |--labels
         |--train
         |--test
      |--visible
         |--train
         |--test
      |--qualities
         |--train
         |--test
      |--corruption
         |--dual_modalilty
            |--ir_rgb_5
            |--ir_rgb_10
            |--ir_rgb_15
            |--ir_rgb_25
            |--ir_rgb_35
            |--ir_rgb_45
            |--ir_rgb_50
            |--ir_zero_100
            |--rgb_zero_100
            |--day_rgb_missing
            |--night_ir_missing
  ```
  
  Edit the paths in `./data/multispectral/FLIR.yaml`，  `./data/multispectral/kaist.yaml`, `./data/multispectral/LLVIP.yaml`  to the proper ones.


### Training and Test
To train and test the model, simply run train.py and test.py respectively. The program was developed and run in a PyCharm environment on a computer with the Windows 10 operating system.

### Evalutaion Result
<div align="center">  Table 1.  Evaluation results of mAP50 on the FLIR, KAIST and LLVIP datasets and their corrupted variants. 
  
| dataset      | 0:0          | 100：0      | 0：100            | Night: T-Missing | Day: RGB-Missing |
|:------------:|:------------:|:-----------:|:-----------------:|:---------------:|:------------:|
| FLIR         | 82.2         | 62.5        | 80.4              | 50.8            | 78.8         | 
| KAIST        | 77.5         | -           | -                 | 36.2            | 68.8         | 
| LLVIP        | 97.4         | -           | -                 | 90.5            | 98.5         |
  
</div>


# Files
**Note**: This is the txt files for evaluation. We continuously optimize our codes, which results in the difference in detection performance. However, the codes of module for multispectral object detection still remain consistent with the methods proposed in this paper.




