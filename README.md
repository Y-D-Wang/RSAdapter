# RSAdapter

The official PyTorch implementation of the paper "[RSAdapter: Adapting Multimodal Models for Remote Sensing Visual Question Answering](https://arxiv.org/pdf/2310.13120.pdf)".

If you find our work useful in your research, please cite:

```
@article{wang2023rsadapter,
  title={RSAdapter: Adapting Multimodal Models for Remote Sensing Visual Question Answering},
  author={Wang, Yuduo and Ghamisi, Pedram},
  journal={arXiv preprint arXiv:2310.13120},
  year={2023}
}
```


## Introduction

In this work, we introduce a novel
method known as RSAdapter, which prioritizes runtime and
parameter efficiency. RSAdapter comprises two key components:
the Parallel Adapter and an additional linear transformation
layer inserted after each fully connected (FC) layer within the
Adapter. This approach not only improves adaptation to pretrained multimodal models but also allows the parameters of the
linear transformation layer to be integrated into the preceding
FC layers during inference, reducing inference costs.

![](Figure/Flowchart.png)


### Preparation

- Download the [RSVQA](https://github.com/syvlo/RSVQA) and [RSIVQA](https://github.com/nikhilrane-21/RSIVQA) datasets. 

### Training


* for RSVQA-LR dataset
	- Change the default path of image files

```shell
python train_lr.py

```
* for RSVQA-HR dataset
	- Change the default path of image files

```shell
python train_hr.py

```
* for RSIVQA dataset
	- Change the default path of image files

```shell
python train_rsi.py

```

- RSAdapter is implemented in https://github.com/Y-D-Wang/RSAdapter/blob/6a7627833ca4daac00eab4b3fdf5ad0a543c5a79/src/t/src/transformers/models/vilt/modeling_vilt_test.py#L478.
- RSadapter is added to the vilt model in https://github.com/Y-D-Wang/RSAdapter/blob/6a7627833ca4daac00eab4b3fdf5ad0a543c5a79/src/t/src/transformers/models/vilt/modeling_vilt_test.py#L546 and https://github.com/Y-D-Wang/RSAdapter/blob/6a7627833ca4daac00eab4b3fdf5ad0a543c5a79/src/t/src/transformers/models/vilt/modeling_vilt_test.py#L561.

### Acknowledgement

The codes are based on [transformers](https://github.com/huggingface/transformers). The authors would also like to thank the contributors to the  [RSVQA](https://github.com/syvlo/RSVQA) and [RSIVQA](https://github.com/nikhilrane-21/RSIVQA) datasets.