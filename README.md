This repo is the implementation of [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP) with AVRET on CSL-daily dataset. Thanks to their great work. 
It currently includes code and pretrained features for the gloss-free SLT task.

## Installation

```bash
conda create -n gfslt python==3.8
conda activate gfslt

# Please install PyTorch according to your CUDA version.
pip install -r requirements.txt
```

## Getting Started

### Preparation
* The pretrain_models of MBart can download from [here](https://pan.baidu.com/s/1BW_7l8-DMqhgxAYOp_ebDA). The extract code is `6f0a`. It includes csl_mbart_char and csl_mbart_char_my.

* The pretrained gloss-free features of CSL-Daily can download from [here](https://pan.baidu.com/s/1PnOt_Tt66TdgN4mLsuzLig). The extract code is `jorx`. And then, put it in the `data/features/` folder.

### Train
```bash
python train_slt.py 
```

## Note
Since GFSLT-VLP is based on the MBart, we did not apply the local clip self-attention (LCSA) module to it.

# LICENSE
The code is released under the MIT license.
