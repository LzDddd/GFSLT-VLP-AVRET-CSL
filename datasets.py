from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
from sacrebleu.metrics import BLEU, CHRF, TER
import os
import random
import numpy as np
import lmdb
import io
import time
import yaml
from vidaug import augmentors as va
from augmentation import *
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from loguru import logger

# global definition
from definition import *


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x


class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip


class S2T_Dataset(Dataset.Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer

        self.list = [key for key, value in self.raw_data.items()]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        name_sample = sample['name']

        tgt_sample = sample['char_text']   # text
        length = sample['new_length']
        img_sample = sample['feature']

        return name_sample, img_sample, tgt_sample, length

    def collate_fn(self, batch):

        name_batch, img_tmp, tgt_batch, src_length_batch = [], [], [], []

        for name, sign, text, sign_lengths in batch:
            name_batch.append(name)
            img_tmp.append(sign)
            tgt_batch.append(text)
            src_length_batch.append(sign_lengths)

        src_input = {}
        src_input['name'] = name_batch
        src_input['text'] = tgt_batch

        src_input['input_ids'] = pad_sequence(img_tmp, padding_value=PAD_IDX, batch_first=True)
        src_input['src_length_batch'] = torch.tensor(src_length_batch)

        new_src_lengths = (((src_input['src_length_batch'] - 5 + 1) / 2) - 5 + 1) / 2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()

        src_input['attention_mask'] = img_padding_mask
        src_input['new_src_length_batch'] = new_src_lengths

        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)

            new_tgt_input = []
            for idx in range(tgt_input['input_ids'].shape[0]):
                temp = []
                for cont in tgt_input['input_ids'][idx, :]:
                    if cont != SPACE_IDX and cont != PAD_IDX:
                        temp.append(cont)
                new_tgt_input.append(torch.stack(temp))
            new_temp = torch.ones((len(new_tgt_input), max([len(i) for i in new_tgt_input])), dtype=tgt_input['input_ids'].dtype)
            for idx, cont in enumerate(new_tgt_input):
                new_temp[idx, :len(cont)] = cont
            tgt_input['input_ids'] = new_temp
            tgt_input['attention_mask'] = (new_temp != PAD_IDX).long()

        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type,
                                              random_shuffle=self.args.random_shuffle, is_train=(self.phase == 'train'))

            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding=True, truncation=True)
                new_masked_tgt_input = []
                for idx in range(masked_tgt_input['input_ids'].shape[0]):
                    temp = []
                    for jdx, cont in enumerate(masked_tgt_input['input_ids'][idx, :]):
                        if cont != SPACE_IDX and cont != PAD_IDX:
                            temp.append(cont)
                    new_masked_tgt_input.append(torch.stack(temp))
                new_temp = torch.ones((len(new_masked_tgt_input), max([len(i) for i in new_masked_tgt_input])),
                                      dtype=masked_tgt_input['input_ids'].dtype)
                for idx, cont in enumerate(new_masked_tgt_input):
                    new_temp[idx, :len(cont)] = cont
                masked_tgt_input['input_ids'] = new_temp
                masked_tgt_input['attention_mask'] = (new_temp != PAD_IDX).long()
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
