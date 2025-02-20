# PyTorch implementation of [Learned Image Transmission with Hierarchical Variational Autoencoder](https://openreview.net/forum?id=UO0bYZdkou) 

This repository is built upon [QARV](https://github.com/duanzhiihao/lossy-vae) and [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22), thanks very much!


## Citation 
``` bash
@article{zhang2024learned,
  title={Learned Image Transmission with Hierarchical Variational Autoencoder},
  author={Zhang, Guangyi and Li, Hanlei and Cai, Yunlong and Hu, Qiyu and Yu, Guanding and Zhang, Runmin},
  booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/HJSCC.git
cd HJSCC/
```

## Requirements
`Python 3.9.21` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt 
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page](https://pytorch.org/get-started/previous-versions/).

## Pretrained Models
- Two pre-trained models are provided in the `HJSCC_feedback/checkpoint` folder.

## Prepare Datasets for Training and Evaluation
- Edit `HJSCC_feedback/config.py` such that
```
train_data_dir = ['/path/to/train/datasets']
test_data_dir  = ['/path/to/test/datasets']
```

## Usage
Example of test the HJSCC_feedback model:
```bash
python main.py --phase test  --checkpoint path_to_checkpoint 
```

