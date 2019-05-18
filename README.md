# LEDNet
This is an unofficial implement of  [LEDNet](https://arxiv.org/abs/1905.02423). 

> the official version：[LEDNet-official](https://github.com/xiaoyufenfei/LEDNet)

## Environment

- Python 3.6
- PyTorch 1.1

## Performance

- Base Size 1024, Crop Size 768, only fine.

| Model  | Paper | OHEM | Epoch | val (crop)  |     val     |
| :----: | :---: | :--: | :---: | :---------: | :---------: |
| LEDNet |   /   |  ✗   |  240  | 44.67/91.85 | 49.79/91.31 |
| LEDNet |   /   |  ✗   | 1000  | 53.77/93.45 | 59.04/93.27 |

- Base Size 1356, Crop Size 1024, only fine.



> The paper only provide the test results: 69.2/86.8 (class mIoU/category mIoU)
>
> - reference the [Fast-SCNN](), we choose epoch=1000



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
$ python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py [--dataset citys] [--batch-size 8] [--base-size 1024] [--crop-size 768] [--epochs 240] [--warmup-factor 0.1] [--warmup-iters 200] [--log-step 10] [--save-epoch 40] [--lr 0.0001]
```

## Prepare data

Your can reference [gluon-cv-cityspaces](https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html#sphx-glr-build-examples-datasets-cityscapes-py) to prepare the dataset

