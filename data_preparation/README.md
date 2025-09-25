# RWKVFusion Data Preparation

We build the mask and captioning data based on Florence2 and SAM2 models.

## Installation
To install the SAM2 model, please run
```bash
cd data_preparation/florence-sam/segment-anything-2
pip install -e .
```
More information, please check the SAM2 [INSTALL.md](florence-sam/segment-anything-2/INSTALL.md)

For hugginface `transformers` package, please install the old version, by running
```bash
pip install transformers==4.45.1
```
otherwise, the script will fail.


## Pretrained Models
For fast running the scripts, the checkpoints are provided in:
- SAM2 model: [[Huggingface]](https://huggingface.co/facebook/sam2.1-hiera-base-plus/tree/main)
Download _`sam2_hiera_base_plus.pt`_ model checkpoint and put it into the `checkpoints/` folder.
- Florence2 model: the huggingface AutoModel class will download the model automatically.


## Start Preparing the Mask and Captions
For the mask prepration, please run
```bash
cd data_preparation/florence-sam
python run_auto_masking.py --img_dir <img_dir> --dataset <dataset_name>
```
Then the mask will be saved in `results/masks/` directory.

For the captions prepration, please run
```bash
cd data_preparation/florence-sam
python run_auto_captioning.py --img_dir <img_dir> --dataset <dataset_name>
```
Then the captions will be saved in `results/captions/` directory as `.csv` files.

## Caption Embeddings
To get the embeddings of the captions, please run
```bash
cd data_preparation/text_process_pipe
python t5_pipe.py --csv_path <csv_path> --save_dir <save_dir>
```
This will generate the embeddings in `.safetensors` format for fast memory loading.
