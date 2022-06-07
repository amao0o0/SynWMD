# SynWMD
Syntax-aware Word Mover’s Distance for Sentence Similarity Modeling

SynWMD, an improved Word Mover's Distance, leverages sentence structural information to improve WMD for sentence similarity modeling. SynWMD incorporates the syntactic dependency parse tree into both the word flow assignment process and the word distance modeling process in WMD to improve the performance on sentence similarity modeling.

# Environment Setup

Linux OS is recommended

```
# create a new virtual environment with conda 
conda create -n synwmd python=3.9.7

# activate
conda activate synwmd

# essentials installation with pip
pip install -r requirements.txt

# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# Data Download
```
cd data/downstream/
bash get_transfer_data.bash
cd ../..
```

# The Results in Semantic Textual Similarity (STS)

Embeddings        | Methods | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | Avg.
------------------|---------|-------|-------|-------|-------|-------|-------|-----
BERT (First-last) |WMD<sub>l2</sub>| 53.03 | 58.96 | 56.79 | 72.11 | 63.56 | 61.01 | 60.91
BERT (First-last) |WMD<sub>cos</sub>| 55.38 | 58.51 | 56.93 | 72.81 | 64.47 | 61.80 | 61.65
BERT (First-last) |WRD| 49.93 | 63.48 | 57.63 | 72.04 | 64.11 | 61.92 | 61.52
BERT (First-last) |WMD<sub>l2</sub>+IDF | 61.19 | 68.67 | 63.72 | 76.87 | 70.16 | 69.56 | 68.36
BERT (First-last) |WMD<sub>cos</sub>+IDF | 63.79 | 69.25| 64.51 | 77.58 | 71.7 | 70.69 | 69.59
BERT (First-last) |BERTScore| 61.32 | 73.00 | 66.52 | 78.47 | 73.43 | 71.77 | 70.75
BERT (First-last) |SynWMD<sub>dwf</sub> | 66.34 | 77.08 | 68.96 | **79.13** | 74.05 | 74.06 | 73.27 
BERT (First-last) |SynWMD<sub>dwf+dwd</sub> | **66.74** | **79.38** | **69.76** | 78.77 | **75.52** | **74.81** | **74.16**
SimCSE-BERT |Sent. Emb. | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85 | 76.92
SimCSE-BERT |WMD<sub>l2</sub>| 64.66 | 79.72 | 73.12 | 81.25 | 76.69 | 77.53 | 75.50
SimCSE-BERT |WMD<sub>cos</sub>| 65.43 | 80.00 | 73.35 | 81.21 | 76.97 | 77.18 | 75.69
SimCSE-BERT |WRD| 64.80 | 80.97 | 74.13 | 80.71 | 76.68 | 78.47 | 75.96
SimCSE-BERT |WMD<sub>l2</sub>+IDF | 67.35 | 81.36 | 74.56 | 82.29 | 78.12 | 79.18 | 77.14
SimCSE-BERT |WMD<sub>cos</sub>+IDF | 68.47 | 81.76 | 74.98 | 82.30 | 78.29 | 78.98 | 77.46
SimCSE-BERT |BERTScore| 66.31 | 82.87 | 75.66 | 83.14 | 79.16 | 80.03 | 77.86
SimCSE-BERT |SynWMD<sub>dwf</sub> | 70.20 | 83.36 | 76.17 | 83.16 | 78.81 | **80.02** | 78.62
SimCSE-BERT |SynWMD<sub>dwf+dwd</sub> | **70.27** | **83.44** | **76.19** | **83.21** | **78.83** | 79.98 | **78.66**

- First ```run parser_download.py``` to download the parser.

- The kernel and bias for whitening BERT (first_last) embedding are provided in `/data/whiten/`. You can also run ```python whiten.py``` to get the whitening kernel and bias for BERT (first_last) embedding.

- The results can be obtained by the demo `eva_stsall_demo.py`
    - Directly running ```python eva_stsall_demo.py``` gives the result of SynWMD<sub>dwf+dwd</sub> using BERT (First-last) embeddings
    - Directly running ```python eva_stsall_demo2.py``` gives the result of SynWMD<sub>dwf+dwd</sub> using SimCSE-BERT embeddings
    - Other results can be reproduced by changing the parameters in the demo `eva_stsall_demo.py` or `eva_stsall_demo2.py`

# Citation
If you find our model is useful in your research, please consider cite our paper: ...

# Acknowledge
A part of the code is modified from other work. Many thanks for

1. [MoverScore](https://github.com/AIPHES/emnlp19-moverscore)
2. [SentEval](https://github.com/facebookresearch/SentEval) Sentence Evluation toolkit.
