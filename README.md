<div align="center">
	
# 🧩 SegRWKV: Linear-Complexity RWKV for Efficient 3D Medical Image Segmentation

</div>

## The overall architecture of SegRWKV

![image](./Pictures/SegRWKV.png)

| Cross-dimensional Feature Enhancement and Cross-dimensional Feature Integration. | X-Shift and Tri-Directional Flatten. |
|---|---|
| ![CFI_CFE](./Pictures/CFI_CFE.png) | ![X_shift_TriDF](./Pictures/X_shift_TriDF.png) |

> Example Results on BraTS2024.

![image](./Pictures/Results_BraTS_2023.png)

> Visual Results on BraTS2023.

![image](./Pictures/visual_BraTS_2023_01.png)

## ⚡ Data downloading

### BraTS 2023 and BraTS 2024

Data of BraTS 2023 is from [https://www.synapse.org/Synapse:syn51156910/wiki/621282](https://www.synapse.org/Synapse:syn51156910/wiki/621282)

Data of BraTS 2024 is from [https://www.synapse.org/Synapse:syn53708249/wiki/626323](https://www.synapse.org/Synapse:syn53708249/wiki/626323)

The BraTS 2023/2024 structure will be in this format:

<table style="width: 100%; table-layout: fixed;">
  <thead>
    <tr>
      <th align="center">BraTS 2023</th>
      <th align="center">BraTS 2024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
<pre>
data/
└── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
    ├── BraTS-GLI-00000-000/
    │   ├── BraTS-GLI-00000-000-seg.nii.gz
    │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   └── BraTS-GLI-00000-000-t2w.nii.gz
    ├── BraTS-GLI-00002-000/
    │   └── ...
    ├── BraTS-GLI-00003-000/
    │   └── ...
    └── ...
</pre>
      </td>
      <td valign="top">
<pre>
data/
└── BraTS2024-BraTS-GLI-TrainingData/
    ├── BraTS-GLI-00005-100/
    │   ├── BraTS-GLI-00005-100-seg.nii.gz
    │   ├── BraTS-GLI-00005-100-t1c.nii.gz
    │   ├── BraTS-GLI-00005-100-t1n.nii.gz
    │   ├── BraTS-GLI-00005-100-t2f.nii.gz
    │   └── BraTS-GLI-00005-100-t2w.nii.gz
    ├── BraTS-GLI-00005-101/
    │   └── ...
    ├── BraTS-GLI-00006-100/
    │   └── ...
    └── ...
</pre>
      </td>
    </tr>
  </tbody>
</table>

### AMOS 2022
 
Data is from [https://amos22.grand-challenge.org/](https://amos22.grand-challenge.org/)

The data structure will be in this format:

```text
data/
└── amos22/
    ├── imagesTr/
    │   ├── amos_0001.nii.gz
    │   ├── amos_0004.nii.gz
    │   ├── amos_0005.nii.gz
    │   └── ...
    ├── imagesVal/
    │   └── ...
    ├── labelsTr/
    │   ├── amos_0001.nii.gz
    │   ├── amos_0004.nii.gz
    │   ├── amos_0005.nii.gz
    │   └── ...
    ├── labelsVal/
    │   └── ...
    ├── dataset.json
    └── readme.md
```

### MSD Task01-Task10

Data is from [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

The Task01-Task10 structure is similar to AMOS 2022.

For the needs of the experiment, we need to organize Task01_BrainTumour into the following data structure (similar to BraTS 2023/2024).

```text
data/
└── Task01_BrainTumour/
    ├── BRATS_001/
    │   ├── img.nii.gz
    │   └── seg.nii.gz
    ├── BRATS_002/
    │   ├── img.nii.gz
    │   └── seg.nii.gz
    ├── BRATS_003/
    │   ├── img.nii.gz
    │   └── seg.nii.gz
    ├── BRATS_004/
    │   └── ...
    ├── BRATS_005/
    │   └── ...
    └── ...
```

## ⚡ Preprocessing, training, and testing

### Brain Tumour - BraTS2023，BraTS2024 and MSD Task01

#### Preprocessing
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

### Other organs - AMOS 2022 and MSD Task02-Task10

#### Preprocessing



