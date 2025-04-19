# MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra

<img src="imgs/model.jpg" alt="model" style="zoom: 35%;" />

Implementation for paper: MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra

This is the code for the ICLR'25 Paper: [MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra](https://openreview.net/forum?id=xJDxVDG3x2).


## Usage

**Download dataset:** You can find and download the processed QM9S dataset named "qm9sp.zip" using this [link](https://osf.io/wrvcn/?view_only=04fe1a045a06457e96e023de12c87072). Select `Files` to access the dataset. Unzip the file into the target folder, and update the `dataset_root` configuration in `examples/ET-QM9-QM9SP-PT.yaml` and `examples/ET-MD17-QM9SP-PT.yaml` to point to the target folder.

**Pre-train and fine-tune:** Please refer to the scripts in the `scripts/` directory for pre-training and fine-tuning.

## Requirements

* Python >= 3.10
* torch>=2.3.1
* torch_cluster>=1.6.3
* torch_geometric>=2.6.1
* torch_scatter>=2.1.2
* ase>=3.23.0
* h5py>=3.11.0
* matplotlib>=3.10.0
* numpy>=1.26.3
* pytorch_lightning>=1.3.8
* PyYAML>=5.4.1
* tqdm>=4.66.5

## Citation

Please cite our paper if you use the code:

```
@inproceedings{wang2025molspectra,
  author       = {Liang Wang and Shaozhen Liu and Yu Rong and Deli Zhao and Qiang Liu and Shu Wu and Liang Wang},
  title        = {MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra},
  booktitle    = {ICLR},
  year         = {2025}
}
```
