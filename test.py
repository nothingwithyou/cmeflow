"""
@Time ： 2024/5/21 19:51
@Auth ： cqy
@File ：test.py
@IDE ：PyCharm
@Motto：salt fish
"""
import torch
import argparse
import numpy as np
import os
from tqdm import *
import torchvision
torchvision.disable_beta_transforms_warning()
from cmeflow.UTFflow import UTFlow
from glob import glob
from utils.flow_viz import flow_to_image
from torchvision import transforms
from PIL import Image
import cv2 as cv
import seaborn as sns
from matplotlib import pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoints', default='checkpoints/step_112000.pth', type=str,
                        help='where to load the test models')
    parser.add_argument('--image_size', default=[256, 256], type=int, nargs='+',        # [720, 160]
                        help='image size for training')
    parser.add_argument('--data_max', default=[6649.7251], type=float, nargs='+',  # [720, 160]
                        help='image max value')
    parser.add_argument('--data_min', default=[0], type=float, nargs='+',  # [720, 160]
                        help='image min values')
    parser.add_argument('--test_dir', default='halo/f1024/22644311', type=str, nargs='+',
                        help='test dataset')
    parser.add_argument('--results', default='results', type=str, nargs='+',
                        help='results dataset')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--seed', default=250, type=int)

    # CMEFlow model
    parser.add_argument('--num_scales', default=2, type=int,
                        help='basic cmeflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2, 8], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1, 4], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1, 1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    return parser


def imnorm(im, mi=0,mx=0, mode=0):
    #   图像最大最小归一化 0-1
    if mi==0 and mx == 0:
        mi, mx = np.min(im), np.max(im)
    im2 = (im - mi) / (mx - mi)
    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0
    return im2


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args([])
    device = torch.device('cuda')
    model = UTFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoints, map_location='cuda')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.module.load_state_dict(weights)
    a = sorted(glob('{}/*.pt'.format(args.test_dir)))
    img_trans = transforms.Compose([
        # rand_crop,
        transforms.Resize(args.image_size, antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    model.eval()
    # model.train()
    datamax = [args.data_max]  # [7979,  213,  333]
    datamin = [args.data_min]
    flow_list = []
    with torch.no_grad():
        for i in range(len(a) - 1):
            img1_path = a[i]
            img2_path = a[i + 1]
            s1 = torch.load(img1_path)
            s2 = torch.load(img2_path)
            s1 = (s1 - datamin[0]) / (datamax[0] - datamin[0])
            s2 = (s2 - datamin[0]) / (datamax[0] - datamin[0])
            s1 = torch.unsqueeze(s1, dim=0)
            s2 = torch.unsqueeze(s2, dim=0)
            #         s1 = torch.unsqueeze(s1[:,7:], dim=0)
            #         s2 = torch.unsqueeze(s2[:,7:], dim=0)
            img1 = img_trans(s1)
            img2 = img_trans(s2)
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            a1 = model(img1.to(device), img2.to(device),
                       attn_splits_list=args.attn_splits_list,
                       corr_radius_list=args.corr_radius_list,
                       prop_radius_list=args.prop_radius_list, )
            outs = a1['flow_preds']
            flow = outs[-1][0].cpu()
            flow = flow.permute(1, 2, 0)
            flow = flow.detach().cpu().numpy()
            flow_list.append(flow)
    flow_list = np.array(flow_list)
    mask = img1[0, 0, :, :] == -1
    mask = mask.numpy().reshape(1, 256, 256, 1)
    flow_list = flow_list * (1 - mask)
    rad = np.sqrt(np.square(flow_list[:, :, :, 0]) + np.square(flow_list[:, :, :, 1]))
    max_rad = rad.max()
    os.makedirs('img_flow/256_22587036', exist_ok=True)
    yuzhi = 11
    flow_mask = mask[0].astype('uint8')
    flow_mask = 1 - flow_mask
    # norm time
    """
    hh = [i.split('/')[-1].split('.')[0] for i in a]
    data_csv = pd.read_csv('../the_date/2016/m_6.csv', index_col=0)
    data_csv[3] = data_csv['0'].apply(lambda x: os.path.splitext(x)[0])
    data_csv.set_index(3, inplace=True)
    a_time = data_csv.loc[hh]
    a_time[4] = a_time['1'] + ' '+ a_time['2']
    a_time_s = pd.to_datetime(a_time[4]).copy()
    date_list = []
    for i in range(a_time_s.shape[0] - 1):
        date_list.append(a_time_s[i+1] -  a_time_s[i])
    date_list = pd.Series(date_list)
    date_list = date_list / np.timedelta64(1, 's')
    date_list = date_list.to_numpy()
    norm_flow = flow_list / (date_list / 720).reshape(date_list.shape[0],1,1,1)
    rad = np.sqrt(np.square(norm_flow[:,:,:,0]) + np.square(norm_flow[:,:,:,1]))
    """
    sns.set_style('white')
    for i in range(flow_list.shape[0]):
        flow = flow_list[i]
        if rad[i].max() > yuzhi:
            flow[rad[i] > yuzhi] = flow[rad[i] > yuzhi] / (rad[i][rad[i] > yuzhi] / yuzhi).reshape(-1, 1)
        flow = cv.GaussianBlur(flow, (5, 5), 0)
        color = flow_to_image(flow, yuzhi)
        img = Image.fromarray(color * flow_mask)
        img.save('{}/{:03d}.png'.format(args.results, i))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)
        plt.imshow(rad[i], vmax=11)
        plt.savefig('{}/wcai_{:03d}.png'.format(args.results, i), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()