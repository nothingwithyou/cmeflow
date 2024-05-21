import torch
import flowiz as fz
import numpy as np
from munch import Munch
from PIL import Image
import random
import argparse
import glob
import json
from munch import Munch
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np

from data.datasets import build_train_dataset
from glob import glob
from torchvision import transforms


class DefultCME(torch.utils.data.Dataset):
    def __init__(self,
                 filenames,
                 transform=None,):
        """
        The dataset read the images on the fly
        Args:
            data (string list): A list of training image path
            targets: A list of test image path
            train: True train mode, false eval mode (print data path)
        """
        self.data = filenames
        self.transform = transform
        self.datamax = [7979,  213,  333]
        self.datamin = [980, -336, -307]
        # Sequence length of each training sample
        # Number of data samples

    def __len__(self):
        return len(self.data)

    def channel_norm(self, img):
        for i in range(len(self.datamax)):
            img[i] = (img[i] - self.datamin[i]) / (self.datamax[i] - self.datamin[i])
        return img

    def __getitem__(self, idx):
        # Handle idx which exceed the length limit
        img1_path = self.data[idx]
        s1 = torch.FloatTensor(np.load(img1_path))
        img1 = self.channel_norm(s1)
        if self.transform:
            img1 = self.transform(img1)

        data_dict = Munch(img1=img1, name=img1_path)
        return data_dict


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_dir', default='tmp', type=str,
                        help='where to load the training data')
    parser.add_argument('--image_size', default=[576, 128], type=int, nargs='+',
                        help='image size for training')
    return parser


def get_args_parser_my():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--image_size', default=[576, 128], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--sequence_length', default=5, type=int, help='the input image sequence length')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--train_dir', default='halo/halo55', type=str, nargs='+',
                        help='train dataset')
    parser.add_argument('--val_dir', default='', type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_workers', default=40, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=500, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=200, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # CMEflow model
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
    parser.add_argument('--lambda_photowarp', type=float, default=2,
                        help='Weight for Bidirectional PHOTOMETRIC Loss')
    parser.add_argument('--lambda_biflow', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_smooth', type=float, default=1,
                        help='Weight for flow smooth loss')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


if __name__ == '__main__':
    parser = get_args_parser_my()
    args = parser.parse_args()

    train_dataset = build_train_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)
    img_trans = transforms.Compose([
        # rand_crop,
        transforms.Resize([1440, 320], antialias=True),
        transforms.Resize([576, 128], antialias=True),
        transforms.Normalize(mean=0.5,
                             std=0.5),
    ])
    fetch = iter(train_loader)
    print(len(train_loader))
    a = 0
    for i in range(len(train_loader)):
        inputs = next(fetch)
        img1, img2, fn = inputs['img1'], inputs['img2'], inputs['name']
        for N in range(args.batch_size):
            for s in range(args.sequence_length):
                if torch.isnan(img1[N][s]).any() or torch.isnan(img2[N][s]).any():
                    print(fn[N] + str(s))
                    print('have none')
                    a += 1

    print(a)
    