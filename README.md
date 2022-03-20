# Deep Learning for Time Series forecasting
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic) ![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic) ![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic) ![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

&ensp; This repo mainly focuses on the progress of time series forecasting using deep learning. It was originally [a collection of models](#models-currently-supported) (transformers, attention models, GRUs) that I used in financial market forecasting, which has been organized into a unified framework for easier use.

&ensp; For beginners, we recommend you read this [paper](https://arxiv.org/abs/2004.13408) or [the brief introduction](/What%20you%20need%20know%20before%20starting%20the%20project.pdf) we provided to learn about time series forecasting.


## Requirements

- Python 3.7
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.7.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Get Started
1. Download data provided by the [repo](https://github.com/thuml/Autoformer). You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). __All the datasets are well pre-processed__ and can be used easily.
2. Train the model. We provide the experiment scripts of all benchmarks under the folder ./scripts. You can reproduce the experiment results by:
```bash
bash ./scripts/ETT_script/Autoformer_ETTm1.sh
bash ./scripts/ECL_script/Autoformer.sh
bash ./scripts/Exchange_script/Autoformer.sh
bash ./scripts/Traffic_script/Autoformer.sh
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/ILI_script/Autoformer.sh
```
Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main.py --model informer --data ETTm1 --attn prob --freq t
```
## Usage on customized data
&ensp;**To run on your customized data**, a `DataSet` class must be provided in `dataloader.py`, then add the `Dataset` to `Exp_Basic.py`. Need to be noted that elements ejected from the `DataSet` class must conform to the model's requirement.

<span id="colablink">See Colab Examples for detail:</span> We provide google colabs to help reproducing and customing our repo, which includes `experiments(train and test)`, `forecasting`, `visualization` and `custom data`.
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC) -->

## Other
Usually we will encounter three forms of data:
1. multi files(usual caused by multi individual) which will cause oom if load all of them. Every separate file contaies train, vail, test.
2. sigle file contaied train, vail, test.
3. multi separate files (usual three) i.e. train, vail, test.



&ensp;For 1, we load a file (train, vail, test)dataset iteratively in a epoch until all files are loaded. For 2, 3, we load  train, vail, test dataset before starting training.

### Models currently supported
We will keep adding series forecasting models to expand this repo.
| Year | Models |Tasks|
| --- | --- |---|
|---|Vanilla Lstm|many to one|
|2014|[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)|many to many|
|2014|[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)|many to many|
|2016|[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)|many to many|
2017|[Attention Is All You Need](https://arxiv.org/abs/1706.03762)|many to many|
|2017|[DeepAR:Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)|many to many|
|2018|[TPA:Temporal Pattern Attention for Multivariate Time Series Forecasting](https://arxiv.org/abs/1809.04206)|many to one|
|2018|[TCN:An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)|many to one|
|2019|[LogTrans:Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/abs/1907.00235)|many to one 
|2020 | [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)|many to many
|2021|[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)|many to many|
||Reformer|many to many|
||Transformer XL|many to many|
||N-BEATS|



## Results on a toy data
## DOING and TODO
1. Add probability estimation function.
2. Improve the network structure(especially attention network) according to our data scenario.
3. Add Tensorboard to record exp.

## Contact
If you have any questions, feel free to contact hyliu through Email (hyliu_sh@outlook.com) or Github issues. 
## Acknowlegements 
To complete the project, we referenced the following repos.
[Informer2020](https://github.com/zhouhaoyi/Informer2020),   [AdjustAutocorrelation](https://github.com/Daikon-Sun/AdjustAutocorrelation), [flow-forecast](https://github.com/hyliush/deep-time-series/tree/master), [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq), [Autoformer](https://github.com/thuml/Autoformer).