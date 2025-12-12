## <div align="center">DRNet: Dynamic Routing for Robust Multispectral Object Detection under Modality Missing</div>

### Introduction
In this paper,  we propose a Dynamic Routing-based Multispectral Object Detection Network (DRNet) that introduces modality awareness and dynamic adaptation mechanisms to address the problem of modality missing in RGB-T networks. The proposed network employs a Modality-Aware Dynamic Feature Enhancement (DFE) module that adaptively routes between multimodal fusion and unimodal enhancement pathways, thereby mitigating the impact of missing modalities. We further introduce a Dynamic Feature Mapping (DFM) module that endows a shared detection head with modality-aware transformations, enabling it to adapt to fused, RGB-only, and thermal-only inputs by capturing modality-specific feature distributions and thereby enhancing cross-modal robustness. In the code, DFE corresponds to MADR in the paper; DFM corresponds to UFH in the paper.

### Installation
Install requirements.txt in a Python>=3.8.0 conda environment.

### Datasets
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




