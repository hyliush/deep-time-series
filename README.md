# Deep Learning for Time Series forecasting
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic) ![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic) ![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic) ![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This repo mainly focuses on the progress of time series forecasting using deep learning. It was originally [a collection of models](#models-currently-supported) (transformers, attention models, GRUs) that I used in financial market forecasting, which has been organized into a unified framework for easier use.

For beginners, we recommend you read this [paper](https://arxiv.org/abs/2004.13408) or [the brief introduction](/) we provided to learn about time series forecasting.


## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

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

**To run on your customized data**, a `DataSet` class must be provided in `dataloader.py`, then add the `Dataset` to `Exp_Basic.py`. Need to be noted that elements ejected from the `DataSet` class must conform to the model's requirement.
## Usage
<span id="colablink">Colab Examples:</span> We provide google colabs to help reproducing and customing our repo, which includes `experiments(train and test)`, `forecasting`, `visualization` and `custom data`.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main.py --model informer --data ETTm1 --attn prob --freq t
```

We provide a more detailed and complete command description for training and testing the model:

```python
python -u main_informer.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --out_size <out_size> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--s_layers <s_layers> --d_ff <d_ff> --factor <factor> --padding <padding>
--distil --dropout <dropout> --attn <attn> --embed <embed> --activation <activation>
--output_hidden --do_predict --mix --cols <cols> --itr <itr>
--num_workers <num_workers> --train_epochs <train_epochs>
--batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_amp --inverse --use_gpu <use_gpu> --gpu <gpu> --use_multi_gpu --devices <devices>
```

### Models currently supported

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


## Contact
If you have any questions, feel free to contact hyliu through Email (hyliu_sh@outlook.com) or Github issues. 
## Acknowlegements 
To complete the project, we referenced the following repos.
[Informer2020](https://github.com/zhouhaoyi/Informer2020),   [AdjustAutocorrelation](https://github.com/Daikon-Sun/AdjustAutocorrelation), [flow-forecast](https://github.com/hyliush/deep-time-series/tree/master), [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq), [Autoformer](https://github.com/thuml/Autoformer).