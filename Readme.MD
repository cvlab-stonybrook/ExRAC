# **[AAAI24] Count What You Want: Exemplar Identification and Few-shot Counting of Human Actions in the Wild**

Our work uses audio cues as exemplar to perform few-shot repetitive action counting in the wild on smartwatch data.
![Teaser](./Img/teaser.jpg)

## Contact
If you have any issues, please contact yifehuang@cs.stonybrook.edu

## Environment set up
```
conda create -n ExRac python=3.8.8
conda activate ExRac
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install cudatoolkit=10.1.243
cd Utils
python setup.py build_ext --inplace
```

## Data and Checkpoints
conda create -n ExRac python=3.8.8
conda activate ExRac
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install omegaconf

## Experiment Reproduction


