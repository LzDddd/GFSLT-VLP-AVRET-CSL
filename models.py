from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
# import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from collections import OrderedDict

import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *
from pathlib import Path
from efficientnet_pytorch import EfficientNet


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size=1024, debug=False, hidden_size=256, num_layers=2, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, src_feats, src_lens, max_len, hidden=None):
        """
        Args:
            - src_feats: (batch_size, max_src_len, 512)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed_emb)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_len)

        return rnn_outputs


class AdaptiveMask(nn.Module):
    """
    DDEM v3
    """

    def __init__(self, input_size=512, output_size=512, dropout=0.1):
        """
        AF module
        :param input_size: dimensionality of the input.
        :param output_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(AdaptiveMask, self).__init__()
        self.lstm = BiLSTMLayer(input_size=input_size, hidden_size=output_size, dropout=dropout)
        self.linear = nn.Linear(output_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-5)

    def forward(self, input_tensor, input_len, k=2, mask=None):
        lstm_o = self.lstm(input_tensor, input_len, input_tensor.shape[1])
        list_out = self.softmax(self.linear(lstm_o).squeeze(-1))
        _, indices = list_out.topk(k, dim=-1, largest=False, sorted=False)
        lstm_o = self.layer_norm(lstm_o)

        # update mask
        if mask is not None:
            sgn_mask_copy = mask.clone().detach()
            for b in range(input_tensor.shape[0]):
                sgn_mask_copy[b, indices[b]] = False
            return lstm_o, sgn_mask_copy
        else:
            return lstm_o


class AdaptiveFusion(nn.Module):
    """
    adaptive Fusion
    """

    def __init__(self, input_size_1=512, input_size_2=512, output_siz=2, bias=False):
        """
        adaptive Fusion instead of normal add
        :param input_size_1:
        :param input_size_2:
        :param output_siz:
        :param bias:
        """
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_siz, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_siz, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)

    def forward(self, input_1, input_2):
        fm_sigmoid = self.sigmoid(self.weight_input_1(input_1) + self.weight_input_2(input_2))
        lambda1 = fm_sigmoid.clone().detach()[:, :, 0].unsqueeze(-1)
        lambda2 = fm_sigmoid.clone().detach()[:, :, 1].unsqueeze(-1)

        fused_output = input_1 + input_2 + torch.mul(lambda1, input_1) + torch.mul(lambda2, input_2)
        fused_output = self.layer_norm(fused_output)
        return fused_output


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return self.dropout(emb + self.pe[:, : emb.size(1)])


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=1,
        skip_connection=True):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.kernel_size = kernel_size
        if type(self.kernel_size)==int:
            conv_1 = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size, stride=1, padding='same')
            conv_2 = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size, stride=1, padding='same')
            self.pwff_layer = nn.Sequential(
                conv_1,
                nn.ReLU(),
                nn.Dropout(dropout),
                conv_2,
                nn.Dropout(dropout),
            )
        elif type(self.kernel_size)==list:
            pwff = []
            first_conv = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size[0], stride=1, padding='same')
            pwff += [first_conv, nn.ReLU(), nn.Dropout(dropout)]
            for ks in kernel_size[1:-1]:
                conv = nn.Conv1d(ff_size, ff_size, kernel_size=ks, stride=1, padding='same')
                pwff += [conv, nn.ReLU(), nn.Dropout(dropout)]
            last_conv = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size[-1], stride=1, padding='same')
            pwff += [last_conv, nn.Dropout(dropout)]

            self.pwff_layer = nn.Sequential(
                *pwff
            )
        else:
            raise ValueError
        self.skip_connection=skip_connection
        if not skip_connection:
            print('Turn off skip_connection in PositionwiseFeedForward')

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_t = x_norm.transpose(1,2)
        x_t = self.pwff_layer(x_t)
        if self.skip_connection:
            return x_t.transpose(1,2)+x
        else:
            return x_t.transpose(1,2)


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.ln = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.ln(x)
        return x


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()


class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits


class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear'):
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model = FeatureExtracter()

        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)

        self.sign_emb = V_encoder(emb_size=1024, feature_size=1024, config=config)

    def forward(self, src_input):
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch'])  # [b, n, c]
        x = self.sign_emb(x)
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output


class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).model.shared.num_embeddings)))

    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=tgt_input['attention_mask'].cuda(),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=masked_tgt_input['attention_mask'].cuda(),
            return_dict=True,
        )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits


class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024):
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt

    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states

    def forward(self, src_input, tgt_input):
        image_features = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype,
                                 requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth


class FeatureExtracter(nn.Module):
    def __init__(self):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet()  # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src, src_length_batch)
        src = self.conv_1d(src)

        return src


class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, attn_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.score = None

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]
            mask = mask[:, None, None, :].float()
            dots -= 10000.0 * (1.0 - mask)
        attn = dots.softmax(dim=-1)
        self.score = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

    def visualize(self):
        return self.score


class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        num_heads = heads
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.score = None

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]
            mask = mask[:, None, None, :].float()
            attn -= 10000.0 * (1.0 - mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.score = attn

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    def visualize(self):
        return self.score


class Cross_att_layer(nn.Module):
    def __init__(self, dim=1024, heads=16, depth=2, dropout=0.1, attn_drop=0.0, mlp_dim=768):
        super(Cross_att_layer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, attn_drop=attn_drop)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, CrossAttention(dim, heads=heads, attn_drop=attn_drop)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))
        self.cls_token_f = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_g = nn.Parameter(torch.randn(1, 1, dim))

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, inp_dim))

    def forward(self, f, fmask, g, gmask):
        B, N, C = f.shape
        cls_token_f = repeat(self.cls_token_f, '() n d -> b n d', b=B)
        f = torch.cat((cls_token_f, f), dim=1)
        cls_token_g = repeat(self.cls_token_g, '() n d -> b n d', b=B)
        g = torch.cat((cls_token_g, g), dim=1)
        for attn1, ff1, c_attn, ff3 in self.layers:
            f = attn1(f) + f
            f = ff1(f) + f

            g = attn1(g) + g
            g = ff1(g) + g

            f_g = c_attn(torch.cat((f[:, 0:1, :], g[:, 1:, :]), dim=1))
            g_f = c_attn(torch.cat((g[:, 0:1, :], f[:, 1:, :]), dim=1))
            f = torch.cat((g_f, f[:, 1:, :]), dim=1)
            g = torch.cat((f_g, g[:, 1:, :]), dim=1)
            f = ff3(f) + f
            g = ff3(g) + g

        return torch.cat((f[:, 0:1, :], g[:, 0:1, :]), dim=1)


class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()

        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                src: Tensor,
                ):

        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0, 2, 1)).permute(0, 2, 1)

        return src


class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter()
        self.mbart = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder'], ignore_mismatched_sizes=True)

        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim, config=config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

        self.mbart_encoder = self.mbart.get_encoder()
        self.adaptive_mask_module1 = AdaptiveMask(input_size=1024, output_size=1024, dropout=0.1)
        self.adaptive_mask_module2 = AdaptiveMask(input_size=1024, output_size=1024, dropout=0.1)
        self.af = AdaptiveFusion(input_size_1=1024, input_size_2=1024)

    def share_forward(self, src_input):
        frames_feature = self.backbone(src_input['input_ids'].cuda())
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):

        inputs_embeds, attention_mask = self.share_forward(src_input)

        _, am_mask1 = self.adaptive_mask_module1(
            input_tensor=inputs_embeds, input_len=src_input['new_src_length_batch'], k=1, mask=attention_mask
        )

        prior_encoder_output_dict = self.mbart_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=am_mask1.cuda(),
            return_dict=True
        )

        lstm2, am_mask2 = self.adaptive_mask_module2(
            input_tensor=prior_encoder_output_dict['last_hidden_state'], input_len=src_input['new_src_length_batch'], k=1, mask=am_mask1
        )
        prior_encoder_output_dict['last_hidden_state'] = self.af(prior_encoder_output_dict['last_hidden_state'],
                                                                 lstm2)

        out = self.mbart(inputs_embeds=inputs_embeds.cuda(),
                         attention_mask=am_mask2.cuda(),
                         encoder_outputs=prior_encoder_output_dict,
                         labels=tgt_input['input_ids'].cuda(),
                         decoder_attention_mask=tgt_input['attention_mask'].cuda(),
                         return_dict=True,
                         )
        output = out['encoder_last_hidden_state'][:, 0, :]

        return out['logits'], output, prior_encoder_output_dict

    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id, prior_encoder_output_dict=None):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(
            encoder_outputs=prior_encoder_output_dict,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id
        )
        return out
