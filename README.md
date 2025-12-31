## <div align="center">DRNet: Dynamic Routing for Robust Multispectral Object Detection under Modality Missing</div>

### Introduction
We propose a Dynamic Routing-based Multispectral Object Detection Network (DRNet), which is evaluated on the FLIR, KAIST, and LLVIP datasets. To address the issue of modality missing in RGB-T networks, DRNet introduces modality awareness and dynamic adaptation mechanisms. The network incorporates a Modality-Aware Dynamic Feature Enhancement (DFE) module, which adaptively routes features between multimodal fusion and unimodal enhancement pathways to reduce the impact of missing modalities. Furthermore, we propose a Dynamic Feature Mapping (DFM) module that equips a shared detection head with modality-aware transformations, allowing it to adapt to fused, RGB-only, and thermal-only inputs by capturing modality-specific feature distributions, thereby improving cross-modal robustness. In our implementation, DFE and DFM correspond to the concepts MADR and UFH introduced in the paper.

### Overview
<div align="center">
  <img src="https://github.com/wliu318/DRNet/diagram" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of DRNet framework </div>
</div>

<div align="center">
  <img src="https://github.com/wliu318/DRNet/FE" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 2. Illustration of the feature enhancement module of MADR module </div>
</div>

### Installation
Install requirements.txt in a Python>=3.8.0 conda environment.

### Datasets
The datasets encompass samples for both training and testing, covering scenarios with complete modalities and those with missing modalities.

- **KAIST**  
Link：https://pan.baidu.com/s/1DY7YkLm0yvO-0osv04m9zg 
Code：axui 

 - **FLIR**  
Link：https://pan.baidu.com/s/1xkdnuPpRwSabuxYmQaT9Ig
Code: kiah 

- **LLVIP** 
Link: https://pan.baidu.com/s/1JzzfX-S5X0zQcP7KWWfyhQ 
Code: emjy 
   
# Files
**Note**: This is the txt files for evaluation. We continuously optimize our codes, which results in the difference in detection performance. However, the codes of module for multimodal object detection still remain consistent with the methods proposed in this paper.




