# PyTorch implementation of [Progressive Learned Image Transmission for Semantic Communication Using Hierarchical VAE](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10907944) 

This repository is built upon [QARV](https://github.com/duanzhiihao/lossy-vae) and [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22), thanks very much!


## Citation 
``` bash
@ARTICLE{zhang2025progressive,
  author={Zhang, Guangyi and Li, Hanlei and Cai, Yunlong and Hu, Qiyu and Yu, Guanding and Qin, Zhijin},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Progressive Learned Image Transmission for Semantic Communication Using Hierarchical {VAE}}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image communication;Image coding;Decoding;Symbols;Probabilistic logic;Training;Receivers;Entropy;Wireless communication;Transmitters;Deep learning;hierarchical variational autoencoder;image transmission;joint source-channel coding;semantic communication},
  doi={10.1109/TCCN.2025.3546935}}
```

## Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/PLIT.git
cd PLIT/
```

## Requirements
`Python 3.9.12` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt 
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page](https://pytorch.org/get-started/previous-versions/).

## Pretrained Models
- Two pre-trained models are provided in the `PLIT_4_phases/checkpoint` folder.

## Prepare Datasets for Training and Evaluation
- Edit `PLIT_4_phases/config.py` such that
```
train_data_dir = ['/path/to/train/datasets']
test_data_dir  = ['/path/to/test/datasets']
```

## Usage
Example of test the PLIT model:
```bash
python main.py --phase test  --checkpoint path_to_checkpoint 
```

