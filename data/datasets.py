import torch
import numpy as np
import glob
import json
import torchvision
import torchvision.transforms.v2 as transforms
torchvision.disable_beta_transforms_warning()


class CMESequence(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 sequence_length,
                 datamax=6649.7251,
                 datamin=0,
                 targets=None,
                 transform=None,
                 train=True):
        """
        The dataset read the images on the fly
        Args:
            data (string list): A list of training image path
            targets: A list of test image path
            train: True train mode, false eval mode (print data path)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.targets = targets
        self.count = 0
        self.train = train
        self.transform = transform
        self.datamax = [datamax]      # [7979,  213,  333]
        self.datamin = [datamin]       # [980, -336, -307]
        # Sequence length of each training sample
        # Number of data samples

    def eval(self):
        self.train = False

    def __len__(self):
        return len(self.data[0])

    def channel_norm(self, img):
        for i in range(len(self.datamax)):
            img[i] = (img[i] - self.datamin[i]) / (self.datamax[i] - self.datamin[i])
        return img

    def __getitem__(self, idx):
        # Handle idx which exceed the length limit
        data_dict = {}
        img1_list = []
        img2_list = []
        name_list = []
        if self.targets:
            label = self.targets[idx]
            data_dict['gt'] = label
        else:
            label = None
            img1_path = self.data[0][idx]
            img2_path = self.data[1][idx]
            img1_name = img1_path.split('/')[-1].replace('.pt', '')
            name_list.append(img1_name)
            # s1 = torch.FloatTensor(torch.load(img1_path))
            # s2 = torch.FloatTensor(torch.load(img2_path))
            s1 = torch.load(img1_path)
            s2 = torch.load(img2_path)
            # s1 = self.channel_norm(s1)
            # s2 = self.channel_norm(s2)
            s1 = (s1 - self.datamin) / (self.datamax - self.datamin)
            s2 = (s2 - self.datamin) / (self.datamax - self.datamin)
            if self.transform:
                s1 = self.transform(s1)
                s2 = self.transform(s2)
        data_dict['img1'] = s1
        data_dict['img2'] = s2
        data_dict['name'] = name_list[0] + '-' + name_list[-1]
        return data_dict


def read_all(data_path):
    # Read the whole dataset
    try:
        img1_name_list = json.load(
            open(data_path + "/img1_name_list.json", 'r'))
        img2_name_list = json.load(
            open(data_path + "/img2_name_list.json", 'r'))
        gt_name_list = []
        try:
            gt_name_list = json.load(open(data_path + "/gt_name_list.json", 'r'))
        except:
            pass
    except:
        data_dir = glob.glob(data_path + "/*")
        print(data_dir)
        gt_name_list = []
        img1_name_list = []
        img2_name_list = []

        for dir in data_dir:
            try:
                gt_name_list.extend(glob.glob(dir + '/*flow.flo'))
            except:
                print('{}: No ground truth file'.format(dir))
            img1_name_list.extend(glob.glob(dir + '/*img1.*'))
            img2_name_list.extend(glob.glob(dir + '/*img2.*'))
        gt_name_list.sort()
        img1_name_list.sort()
        img2_name_list.sort()
        print(gt_name_list[0], img1_name_list[0], img2_name_list[0])
        print(len(gt_name_list), len(img1_name_list), len(img2_name_list))
        assert (len(gt_name_list) == len(img1_name_list))
        assert (len(img2_name_list) == len(img1_name_list))

    return img1_name_list, img2_name_list, gt_name_list


def build_train_dataset(args):
    im1, im2, gt = read_all(args.train_dir)
    img_trans = transforms.Compose([
        # rand_crop,
        transforms.Resize(args.image_size, antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    datasets = CMESequence([im1, im2], args.sequence_length, args.data_max, args.data_min, gt, transform=img_trans)
    return datasets
