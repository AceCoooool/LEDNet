import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch import optim
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
import utils as ptutil
from utils.metric_seg import SegmentationMetric
from data import get_segmentation_dataset
from data.sampler import make_data_sampler, IterationBasedBatchSampler
from model.loss import MixSoftmaxCrossEntropyLoss, OHEMSoftmaxCrossEntropyLoss
from model.lr_scheduler import WarmupPolyLR
from model.lednet import LEDNet


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='LEDNet Segmentation')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--height', type=int, default=512,  # 1024
                        help='base image size')
    parser.add_argument('--width', type=int, default=256,  # 512
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--ohem', type=ptutil.str2bool, default='false',
                        help='whether using ohem loss')
    parser.add_argument('--epochs', type=int, default=240, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--warmup-iters', type=int, default=200,  # 500
                        help='warmup iterations')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--eval-epochs', type=int, default=-1,
                        help='validate interval')
    parser.add_argument('--skip-eval', type=ptutil.str2bool, default='False',
                        help='whether to skip evaluation')
    # cuda and logging
    parser.add_argument('--no-cuda', type=ptutil.str2bool, default='False',
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--log-step', type=int, default=1,
                        help='iteration to show results')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='epoch interval to save model.')
    parser.add_argument('--save-dir', type=str, default=cur_path,
                        help='Resume from previously saved parameters if not None.')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    # the parser
    args = parser.parse_args()

    args.lr = args.lr * args.batch_size
    return args


class Trainer(object):
    def __init__(self, args):
        self.device = torch.device(args.device)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'height': args.height,
                       'width': args.width}
        trainset = get_segmentation_dataset(
            args.dataset, split=args.train_split, mode='train', **data_kwargs)
        args.per_iter = len(trainset) // (args.num_gpus * args.batch_size)
        args.max_iter = args.epochs * args.per_iter
        if args.distributed:
            sampler = data.DistributedSampler(trainset)
        else:
            sampler = data.RandomSampler(trainset)
        train_sampler = data.sampler.BatchSampler(sampler, args.batch_size, True)
        train_sampler = IterationBasedBatchSampler(train_sampler, num_iterations=args.max_iter)
        self.train_loader = data.DataLoader(trainset, batch_sampler=train_sampler, pin_memory=True,
                                            num_workers=args.workers)
        if not args.skip_eval or 0 < args.eval_epochs < args.epochs:
            valset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
            val_sampler = make_data_sampler(valset, False, args.distributed)
            val_batch_sampler = data.sampler.BatchSampler(val_sampler, args.test_batch_size, False)
            self.valid_loader = data.DataLoader(valset, batch_sampler=val_batch_sampler,
                                                num_workers=args.workers, pin_memory=True)

        # create network
        self.net = LEDNet(trainset.NUM_CLASS)

        if args.distributed:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net.to(self.device)
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                self.net.load_state_dict(torch.load(args.resume))
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        # create criterion
        if args.ohem:
            min_kept = args.batch_size * args.crop_size ** 2 // 16
            self.criterion = OHEMSoftmaxCrossEntropyLoss(thresh=0.7, min_kept=min_kept, use_weight=False)
        else:
            self.criterion = MixSoftmaxCrossEntropyLoss()

        # optimizer and lr scheduling
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        self.scheduler = WarmupPolyLR(self.optimizer, T_max=args.max_iter, warmup_factor=args.warmup_factor,
                                      warmup_iters=args.warmup_iters, power=0.9)

        if args.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank], output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.num_class)
        self.args = args

    def training(self):
        self.net.train()
        save_to_disk = ptutil.get_rank() == 0
        start_training_time = time.time()
        trained_time = 0
        tic = time.time()
        end = time.time()
        iteration, max_iter = 0, self.args.max_iter
        save_iter, eval_iter = self.args.per_iter * self.args.save_epoch, self.args.per_iter * self.args.eval_epochs
        # save_iter, eval_iter = 10, 10

        logger.info("Start training, total epochs {:3d} = total iteration: {:6d}".format(self.args.epochs, max_iter))

        for i, (image, target) in enumerate(self.train_loader):
            iteration += 1
            self.scheduler.step()
            self.optimizer.zero_grad()
            image, target = image.to(self.device), target.to(self.device)
            outputs = self.net(image)
            loss_dict = self.criterion(outputs, target)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = ptutil.reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()
            trained_time += time.time() - end
            end = time.time()
            if iteration % args.log_step == 0:
                eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
                log_str = ["Iteration {:06d} , Lr: {:.5f}, Cost: {:.2f}s, Eta: {}"
                               .format(iteration, self.optimizer.param_groups[0]['lr'], time.time() - tic,
                                       str(datetime.timedelta(seconds=eta_seconds))),
                           "total_loss: {:.3f}".format(losses_reduced.item())]
                log_str = ', '.join(log_str)
                logger.info(log_str)
                tic = time.time()
            if save_to_disk and iteration % save_iter == 0:
                model_path = os.path.join(self.args.save_dir, "{}_iter_{:06d}.pth"
                                          .format('LEDNet', iteration))
                self.save_model(model_path)
            # Do eval when training, to trace the mAP changes and see performance improved whether or nor
            if args.eval_epochs > 0 and iteration % eval_iter == 0 and not iteration == max_iter:
                metrics = self.validate()
                ptutil.synchronize()
                pixAcc, mIoU = ptutil.accumulate_metric(metrics)
                if pixAcc is not None:
                    logger.info('pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))
                self.net.train()
        if save_to_disk:
            model_path = os.path.join(self.args.save_dir, "{}_iter_{:06d}.pth"
                                      .format('LEDNet', max_iter))
            self.save_model(model_path)
        # compute training time
        total_training_time = int(time.time() - start_training_time)
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
        # eval after training
        if not self.args.skip_eval:
            metrics = self.validate()
            ptutil.synchronize()
            pixAcc, mIoU = ptutil.accumulate_metric(metrics)
            if pixAcc is not None:
                logger.info('After training, pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))

    def validate(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        self.metric.reset()
        torch.cuda.empty_cache()
        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
            model = self.net.module
        else:
            model = self.net
        model.eval()
        tbar = tqdm(self.valid_loader)
        for i, (image, target) in enumerate(tbar):
            # if i == 10: break
            image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                outputs = model(image)
            self.metric.update(target, outputs)
        return self.metric

    def save_model(self, model_path):
        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
            model = self.net.module
        else:
            model = self.net
        torch.save(model.state_dict(), model_path)
        logger.info("Saved checkpoint to {}".format(model_path))


if __name__ == "__main__":
    args = parse_args()

    # device setting
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    if not args.no_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)

    args.lr = args.lr * args.num_gpus  # scale by num gpus

    logger = ptutil.setup_logger('Segmentation', args.save_dir, ptutil.get_rank(), 'log_seg.txt', 'w')
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    trainer = Trainer(args)

    trainer.training()
    torch.cuda.empty_cache()
