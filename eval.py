import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
import utils as ptutil
from model.lednet import LEDNet
from data import get_segmentation_dataset
from data.sampler import make_data_sampler
from utils.metric_seg import SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Segmentation.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='Select dataset.')
    parser.add_argument('--split', type=str, default='val',
                        help='Select val|test, evaluate in val or test data')
    parser.add_argument('--mode', type=str, default='testval',
                        help='Select testval|val, w/o corp and with crop')
    parser.add_argument('--height', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--width', type=int, default=512,
                        help='crop image size')

    parser.add_argument('--pretrained', type=str,
                        default='/home/ace/cbb/own/pretrained/seg/lednet/LEDNet_iter_013800.pth',
                        help='Default Pre-trained model root.')

    # device
    parser.add_argument('--cuda', type=ptutil.str2bool, default='true',
                        help='Training with GPUs.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def validate(net, val_data, metric, device):
    net.eval()
    tbar = tqdm(val_data)
    for i, (data, targets) in enumerate(tbar):
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            predicts = net(data)
        metric.update(targets, predicts)
    return metric


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cpu')
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False if args.mode == 'testval' else True
        device = torch.device('cuda')
    else:
        distributed = False

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)

    # Load Model
    model = LEDNet(19)
    model.load_state_dict(torch.load(args.pretrained))
    model.keep_shape = True if args.mode == 'testval' else False
    model.to(device)

    # testing data
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    data_kwargs = {'width': args.width, 'height': args.height, 'transform': input_transform}

    val_dataset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode, **data_kwargs)
    sampler = make_data_sampler(val_dataset, False, distributed)
    batch_sampler = data.BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    val_data = data.DataLoader(val_dataset, shuffle=False, batch_sampler=batch_sampler,
                               num_workers=args.num_workers)
    metric = SegmentationMetric(val_dataset.num_class)

    metric = validate(model, val_data, metric, device)
    ptutil.synchronize()
    pixAcc, mIoU = ptutil.accumulate_metric(metric)
    if ptutil.is_main_process():
        print('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
