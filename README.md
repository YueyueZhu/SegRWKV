<div align="center">
	
# 🧩 SegRWKV: Linear-Complexity RWKV for Efficient 3D Medical Image Segmentation

</div>

## The overall architecture of SegRWKV

> SegRWKV is a hierarchical encoder-decoder network composed of 5×6 cascaded modules.


> Cross-dimensional Feature Enhancement and Cross-dimensional Feature Integration.


> X-Shift and Tri-Directional Flatten.

## Preprocessing, training, testing, inference, and metrics computation

### Data downloading 

Data is from


### Preprocessing
In my setting, the data directory of BraTS2023 is : "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"

First, we need to run the rename process.

```bash 
python 1_rename_mri_data.py
```

Then, we need to run the pre-processing code to do resample, normalization, and crop processes.

```bash
python 2_preprocessing_mri.py
```

After pre-processing, the data structure will be in this format:
