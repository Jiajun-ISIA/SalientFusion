from itertools import product

from random import choice
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop
import pdb

import json


BICUBIC = InterpolationMode.BICUBIC
n_px = 224

def crop_non_white_area(img):
    # 将图像转换为 NumPy 数组
    img_array = np.array(img)
    
    # 找到非白色区域的边界
    mask = np.any(img_array != [255, 255, 255], axis=-1)  # 假设白色为 [255, 255, 255]
    coords = np.argwhere(mask)
    
    # 如果没有非白色区域，则返回原图
    if coords.size == 0:
        return img
    
    # 获取边界坐标
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    
    # 裁剪图像
    cropped_img = img.crop((top_left[1], top_left[0], bottom_right[1] + 1, bottom_right[0] + 1))
    
    return cropped_img


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                crop_non_white_area,
                RandomResizedCrop(n_px),
                # Resize((n_px,n_px)),
                # CenterCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                crop_non_white_area,
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        # pdb.set_trace()
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                crop_non_white_area,
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        # print(img)
        img = Image.open(file).convert('RGB')
        
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            same_prim_sample=False
    ):
        self.real = False


        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.same_prim_sample = same_prim_sample

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')
        self.loader_woback = ImageLoader(self.root + '/images_woback/')
        self.loader_depth = ImageLoader(self.root + '/images_depth/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        if self.phase == 'train' and self.same_prim_sample:
            self.same_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            self.same_obj_diff_attr_dict = {pair: list() for pair in self.train_pairs}
            for i_sample, sample in enumerate(self.train_data):
                sample_attr, sample_obj = sample[1], sample[2]
                for pair_key in self.same_attr_diff_obj_dict.keys():
                    if (pair_key[1] == sample_obj) and (pair_key[0] != sample_attr):
                        self.same_obj_diff_attr_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] == sample_attr):
                        self.same_attr_diff_obj_dict[pair_key].append(i_sample)


    def get_split_info(self):
        # pdb.set_trace()
        if self.real:
            data_load = self.root + '/metadata_compositional-split-natural-real.t7'
            data = torch.load(data_load)
        else:
            data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        # output_file = 'utzap_dataset.json'
        # with open(output_file, 'w', encoding='utf-8') as json_file:
        #     json.dump(data, json_file, ensure_ascii=False, indent=4)
        # exit()
        # pdb.set_trace()
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                # pdb.set_trace()
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        
        if self.real:
             ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs_real.txt' % (self.root, self.split))
        else:
            ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]

        img = self.loader(image)
        img_woback = self.loader_woback(image)
        img_depth = self.loader_depth(image)

        
        img = self.transform(img)
        img_woback = self.transform(img_woback)
        img_depth = self.transform(img_depth)


        if self.phase == 'train':
            data = [
                img, img_woback, img_depth, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, img_woback, img_depth, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        if self.phase == 'train' and self.same_prim_sample:
            # pdb.set_trace()
            [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_attr_img = self.transform(self.loader(same_attr_image))
            same_obj_img = self.transform(self.loader(same_obj_image))
            data += [same_attr_img, self.attr2idx[same_attr], self.obj2idx[diff_obj], 
                     self.train_pair_to_idx[(same_attr, diff_obj)], same_attr_mask,
                     same_obj_img, self.attr2idx[diff_attr], self.obj2idx[same_obj], 
                     self.train_pair_to_idx[(diff_attr, same_obj)], same_obj_mask]

        return data

    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        if phase=='attr':
            candidate_list = self.same_attr_diff_obj_dict[(label_A, label_B)]
        else:
            candidate_list = self.same_obj_diff_attr_dict[(label_B, label_A)]
        if len(candidate_list) != 0:
            idx = choice(candidate_list)
            mask = 1
        else:
            idx = choice(list(range(len(self.data))))
            mask = 0
        return self.data[idx], mask

    def __len__(self):
        return len(self.data)
