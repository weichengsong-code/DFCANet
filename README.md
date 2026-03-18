# DFCANet: A Multimodal Cross-Attention Fusion Network for Text-Image Forgery Detection
Authors: Weicheng Song, Mingliang Gao




## Installation

### Download
```
mkdir code
cd code
git clone https://github.com/rshaojimmy/MultiModal-DeepFake.git
cd MultiModal-DeepFake
```


### Environment
We recommend using Anaconda to manage the python environment:
```
conda create -n DGM4 python=3.8
conda activate DGM4
conda install --yes -c pytorch pytorch=1.10.0 torchvision==0.11.1 cudatoolkit=11.3
pip install -r requirements.txt
conda install -c conda-forge ruamel_yaml
```


## Dataset Preparation




### Prepare data
Download the DGM<sup>4</sup> dataset through this link: [DGM4](https://huggingface.co/datasets/rshaojimmy/DGM4)

Then download the pre-trained model through this link: [ALBEF_4M.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth) (refer to [ALBEF](https://github.com/salesforce/ALBEF))

Put the dataset into a `./datasets` folder at the same root of `./code`, and put the `ALBEF_4M.pth` checkpoint into `./code/MultiModel-Deepfake/`. After unzip all sub files, the structure of the code and the dataset should be as follows:

```
./
├── code
│   └── MultiModal-Deepfake (this github repo)
│       ├── configs
│       │   └──...
│       ├── dataset
│       │   └──...
│       ├── models
│       │   └──...
│       ...
│       └── ALBEF_4M.pth
└── datasets
    └── DGM4
        ├── manipulation
        │   ├── infoswap
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   ├── simswap
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   ├── StyleCLIP
        │   |   ├── ...
        |   |   └── xxxxxx.jpg
        │   └── HFGI
        │       ├── ...
        |       └── xxxxxx.jpg
        ├── origin
        │   ├── gardian
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   ├── usa_today
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   ├── washington_post
        │   |   ├── ...
        |   |   ...
        |   |   └── xxxx
        │   |       ├── ...
        │   |       ...
        │   |       └── xxxxxx.jpg
        │   └── bbc
        │       ├── ...
        |       ...
        |       └── xxxx
        │           ├── ...
        │           ...
        │           └── xxxxxx.jpg
        └── metadata
            ├── train.json
            ├── test.json
            └── val.json

```


## Training

Modify `train.sh` and run:
```
sh train.sh
```

You can change the network and optimization configurations by modifying the configuration file `./configs/train.yaml`.


## Testing
Modify `test.sh` and run:
```
sh test.sh
```

