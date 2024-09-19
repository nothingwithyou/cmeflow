import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import *
import torchvision

from data.datasets import build_train_dataset
from cmeflow.UTFflow import UTFlow
from cmeflow.loss import unsupervised_error
from utils.logger import Logger
from utils import misc


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--image_size', default=[256, 256], type=int, nargs='+',        # [720, 160]
                        help='image size for training')
    parser.add_argument('--data_max', default=[6649.7251], type=float, nargs='+',  # [720, 160]
                        help='image max value')
    parser.add_argument('--data_min', default=[0], type=float, nargs='+',  # [720, 160]
                        help='image min values')

    parser.add_argument('--sequence_length', default=5, type=int, help='the input image sequence length')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--train_dir', default='halo/f1024', type=str, nargs='+',
                        help='train dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--w', default=24, type=int)
    parser.add_argument('--num_workers', default=40, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--grad_clip', default=0.9, type=float)
    parser.add_argument('--num_steps', default=60000, type=int)
    parser.add_argument('--seed', default=250, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=5000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=500, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # CMEFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic cmeflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')
    parser.add_argument('--lambda_photowarp', type=float, default=1,
                        help='Weight for Bidirectional PHOTOMETRIC Loss')
    parser.add_argument('--lambda_biflow', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_smooth', type=float, default=1,
                        help='Weight for flow smooth loss')
    parser.add_argument('--lambda_curl', type=float, default=1,
                        help='Weight for flow curl loss')
    parser.add_argument('--lambda_div', type=float, default=1,
                        help='Weight for flow div loss')
    parser.add_argument('--lambda_ts', type=float, default=1,
                        help='Weight for spatiotemporal loss')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


if __name__ == '__main__':
    torchvision.disable_beta_transforms_warning()
    parser = get_args_parser()
    args = parser.parse_args()
    if not args.eval and not args.submission and args.inference_dir is None:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            print(args)
            misc.save_args(args)
            misc.check_path(args.checkpoint_dir)
            misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = UTFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)
    # model = torch.compile(model)
    # model = torch.compile(LiteFlowNet(args)).to(device)
    if not args.eval and not args.submission and not args.inference_dir:
        print('Model definition:')
        print(model)

        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        # model = torch.compile(model)
        model_without_ddp = model.module
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and not args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']

        print(' start_step: %d' % (start_step))

    # training datset
    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    # Multi-processing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps,  # args.num_steps * args.sequence_length + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    summary_writer = SummaryWriter(args.checkpoint_dir)
    logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                    start_step=start_step)
    fetch = iter(train_loader)
    print('Start training')
    model.train()
    with tqdm(total=(args.num_steps-start_step), ncols=120) as _tqdm:
        for i in range(start_step, args.num_steps):
            try:
                inputs = next(fetch)
            except (AttributeError, StopIteration):
                fetch = iter(train_loader)
                inputs = next(fetch)
            # mannual change random seed for shuffling every epoch
            img1, img2 = inputs['img1'].to('cuda'), inputs['img2'].to('cuda')
            # img1, img2 = \
            #     inputs['img1'].permute(1, 0, 2, 3, 4).contiguous(), inputs['img2'].permute(1, 0, 2, 3, 4).contiguous()

            flow_name = inputs['name']
            out = []
            # train_opt = torch.compile(train_fun)
            train_opt = train_fun
            a1 = model(img1, img2, flow_old=None,
                       attn_splits_list=args.attn_splits_list,
                       corr_radius_list=args.corr_radius_list,
                       prop_radius_list=args.prop_radius_list, )

            a2 = model(img2, img1,
                       attn_splits_list=args.attn_splits_list,
                       corr_radius_list=args.corr_radius_list,
                       prop_radius_list=args.prop_radius_list, )
            flow1 = a1['flow_preds']
            flow2 = a2['flow_preds']
            sloss, loss_dic = unsupervised_error(flow1, flow2, img1, img2, args)
            sloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            # lr_scheduler.step()
            logger.push(loss_dic)
            logger.add_image_summary(img1, img2, flow1)
            _tqdm.set_postfix(loss='{:.4f}'.format(loss_dic.total))
            _tqdm.update(1)

            if (i+1) % args.save_ckpt_freq == 0 or i == args.num_steps:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % (i+1))
                torch.save({
                    'model': model_without_ddp.state_dict()
                }, checkpoint_path)

            if (i+1) % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': i,
                }, checkpoint_path)

                # support validation on multiple datasets
    print('Training done')
