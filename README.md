# LEDNet
This is an unofficial implement of  [LEDNet](https://arxiv.org/abs/1905.02423). 

> the official version：[LEDNet-official](https://github.com/xiaoyufenfei/LEDNet)

## Environment

- Python 3.6
- PyTorch 1.1

## Performance

#### old-version (without dropout) --- will delete

- Base Size 1024, Crop Size 768, only fine. (old-version, without dropout)

| Model  | Paper | OHEM |   lr   | Epoch | val (crop)  |     val     |
| :----: | :---: | :--: | :----: | :---: | :---------: | :---------: |
| LEDNet |   /   |  ✗   | 0.0001 |  240  | 44.67/91.85 | 49.79/91.31 |
| LEDNet |   /   |  ✗   | 0.0001 | 1000  | 53.77/93.45 | 59.04/93.27 |

- Base Size 1356, Crop Size 1024, only fine. (old-version, without dropout)

| Model  | Paper | OHEM |   lr   | Epoch | val (crop)  | val  |
| :----: | :---: | :--: | :----: | :---: | :---------: | :--: |
| LEDNet |   /   |  ✗   | 0.0001 | 1000  | 56.30/93.90 |      |

- Height 1024, Width 512. (new-version)

| Model  | Paper | OHEM | Drop-rate |   lr   | Epoch | val (crop)  |     val     |
| :----: | :---: | :--: | :-------: | :----: | :---: | :---------: | :---------: |
| LEDNet |   /   |  ✗   |    0.1    | 0.0001 |  300  | 39.03/88.60 | 21.17/72.79 |
| LEDNet |   /   |  ✗   |    0.1    | 0.0001 |  800  | 41.70/89.46 |             |

#### new version (with dropout)

- Base Size 1024, Crop Size 768, only fine. (new-version, with dropout)

| Model  | Paper | OHEM | Drop-rate |   lr   | Epoch | val (crop)  |     val     |
| :----: | :---: | :--: | :-------: | :----: | :---: | :---------: | :---------: |
| LEDNet |   /   |  ✗   |    0.1    | 0.0005 |  800  | 60.32/94.51 | 66.29/94.40 |
| LEDNet |   /   |  ✗   |    0.3    | 0.0005 |  800  | 59.30/94.32 | 65.29/94.11 |
| LEDNet |   /   |  ✗   |    0.1    | 0.001  |  400  |             |             |

> The paper only provide the test results: 69.2/86.8 (class mIoU/category mIoU)
>
> - reference the [Fast-SCNN](), we choose epoch=1000 (here we use 800)

TODO: test larger learning rate (in other models, using learning rate 0.1)

##  Demo

TODO



## Evaluation

The default data root is `~/.torch/datasets` (You can download dataset and build a soft-link to it)

```shell
$ python eval.py [--mode testval] [--pretrained root-of-pretrained-model] [--cuda true]
```

## Training

Recommend to using distributed training.

```shell
$ export NGPUS=4
$ python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py [--dataset citys] [--batch-size 8] [--base-size 1024] [--crop-size 768] [--epochs 240] [--warmup-factor 0.1] [--warmup-iters 200] [--log-step 10] [--save-epoch 40] [--lr 0.0005]
```

## Prepare data

Your can reference [gluon-cv-cityspaces](https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html#sphx-glr-build-examples-datasets-cityscapes-py) to prepare the dataset

