import torch
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import mne
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data._utils.collate
import sklearn.metrics
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

from eeg_feat_extractor import FeatureExtractor
from vit_pytorch import vit


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    return path


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, video_feat, all_eeg_signal, person_ids, seg_id=0,
                 window_seconds=3, sep_seconds=1, shift_seconds=1, start_img_index=None, end_img_index=None,
                 fps=25, eeg_sr=1000, device='cpu'):
        """
        需要返回的是一段指定长度的EEG信号，对应的图片流，还有不匹配的相隔至少一定长度的图片流。这两个图片流要打乱顺序，以第一个是否匹配作为标签。
        """
        self.video_feat = video_feat  # [L,E]
        self.video_length = video_feat.shape[0]
        self.all_eeg_signal = all_eeg_signal
        self.person_ids = person_ids
        self.seg_id = seg_id
        self.person_num = len(person_ids)
        self.device = device
        self.window_length = int(window_seconds * eeg_sr)
        self.sr_times = eeg_sr / fps
        self.sep_seconds = sep_seconds
        self.shift_seconds = shift_seconds
        if start_img_index is None:
            self.start_img_index = 0
        else:
            self.start_img_index = start_img_index
        if end_img_index is None:
            self.end_img_index = self.video_length
        else:
            self.end_img_index = end_img_index
        self.sep_length = int(sep_seconds * eeg_sr)
        self.sep_img_length = int(sep_seconds * fps)
        self.seg_img_length = int(window_seconds * fps)
        self.shift_img_length = int(shift_seconds * fps)
        self.fps = fps
        self.eeg_sr = eeg_sr
        self.get_seg_num()

    def __getitem__(self, index):
        person_idx = index // self.seg_num
        seg_index = index % self.seg_num + self.left_start_seg_idx
        img_index = self.shift_img_length * seg_index + self.start_img_index
        person_id = self.person_ids[person_idx]
        seg_start = int(img_index * self.sr_times)
        seg_start = seg_start if seg_start > 0 else 0
        match = self.video_feat[img_index:img_index + self.seg_img_length]
        mismatch_start_index = img_index + self.seg_img_length + self.sep_img_length
        mismatch = self.video_feat[mismatch_start_index:mismatch_start_index + self.seg_img_length]
        mismatch = torch.tensor(mismatch, dtype=torch.float32, device=self.device)
        match = torch.tensor(match, dtype=torch.float32, device=self.device)
        assert match.shape[0] == self.seg_img_length and mismatch.shape[0] == self.seg_img_length, \
            f'mismatch_start_index:{mismatch_start_index},img_index:{img_index},self.video_length:{self.video_length},seg_len:{self.seg_img_length}'
        mismatch = torch.swapdims(mismatch, 0, 1)
        match = torch.swapdims(match, 0, 1)
        video_feat = {}
        video_feat['v0'] = match
        video_feat['v0_idx'] = img_index
        video_feat['v1'] = mismatch
        video_feat['v1_idx'] = mismatch_start_index

        eeg_signal = self.all_eeg_signal[person_id][f'{self.seg_id}'][:, seg_start:seg_start + self.window_length]
        img_index = torch.tensor(img_index, dtype=torch.int64, device=self.device)
        person_id = torch.tensor(person_id, dtype=torch.int, device=self.device)
        person_idx = torch.tensor(person_idx, device=self.device)
        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float32, device=self.device)
        eeg_signal = eeg_signal / eeg_signal.abs().max() * 0.8
        eeg_signal = torch.nn.functional.pad(eeg_signal, (0, self.window_length - eeg_signal.shape[1]))
        item = {
            "img_idx": img_index,  # int
            "person_id": person_id,  # int real
            "person_nnid": person_idx,  # int re-indexed
            "eeg": eeg_signal,  # [64,400]
            **video_feat
        }
        return item

    def get_seg_num(self, ):
        L = self.end_img_index - self.start_img_index
        s = self.shift_img_length
        l = self.seg_img_length
        p = self.sep_img_length
        i_s = max(int((-l - p) / s), 0)
        n = int((L - l - max(p + l, 0)) / s) + 1 - i_s
        self.seg_num = n
        self.left_start_seg_idx = i_s

    def __len__(self):
        return self.seg_num * self.person_num


class TwoSideDataset(torch.utils.data.Dataset):
    def __init__(self, video_feat, all_eeg_signal, person_ids, seg_id=0,
                 window_seconds=3, sep_seconds=1, shift_seconds=1, start_img_index=None, end_img_index=None,
                 fps=25, eeg_sr=1000, device='cpu'):
        """
        这个数据集确保绝对的左右两边mismatch相同数量，且不存在某个match只有一边的mismatch的情况。
        """
        self.video_feat = video_feat  # [L,E]
        self.video_length = video_feat.shape[0]
        self.all_eeg_signal = all_eeg_signal
        self.person_ids = person_ids
        self.seg_id = seg_id
        self.person_num = len(person_ids)
        self.device = device
        self.window_length = int(window_seconds * eeg_sr)
        self.sr_times = eeg_sr / fps
        self.sep_seconds = sep_seconds
        assert sep_seconds > 0, 'Only support positive separation seconds'
        self.shift_seconds = shift_seconds
        if start_img_index is None:
            self.start_img_index = 0
        else:
            self.start_img_index = start_img_index
        if end_img_index is None:
            self.end_img_index = self.video_length
        else:
            self.end_img_index = end_img_index
        self.sep_length = int(sep_seconds * eeg_sr)
        self.sep_img_length = int(sep_seconds * fps)
        self.seg_img_length = int(window_seconds * fps)
        self.shift_img_length = int(shift_seconds * fps)
        self.period_img_length = self.seg_img_length + self.sep_img_length
        self.fps = fps
        self.eeg_sr = eeg_sr
        self.get_seg_num()

    def __getitem__(self, index):
        person_idx = index // self.seg_num
        seg_index = index % self.seg_num + self.left_start_seg_idx
        img_index = self.shift_img_length * seg_index + self.start_img_index  # first mismatch begins
        person_id = self.person_ids[person_idx]
        match_start_img_index = img_index + self.period_img_length
        match = self.video_feat[match_start_img_index:match_start_img_index + self.seg_img_length]
        if np.random.uniform(0, 1, None) > 0.5:  # 有一半的概率选择左右
            mismatch_start_index = img_index
        else:
            mismatch_start_index = img_index + 2 * self.period_img_length
        mismatch = self.video_feat[mismatch_start_index:mismatch_start_index + self.seg_img_length]
        mismatch = torch.tensor(mismatch, dtype=torch.float32, device=self.device)
        match = torch.tensor(match, dtype=torch.float32, device=self.device)
        assert match.shape[0] == self.seg_img_length and mismatch.shape[0] == self.seg_img_length, \
            f'mismatch_start_index:{mismatch_start_index},img_index:{img_index},self.video_length:{self.video_length},seg_len:{self.seg_img_length}'
        mismatch = torch.swapdims(mismatch, 0, 1)
        match = torch.swapdims(match, 0, 1)
        video_feat = {}
        video_feat['v0'] = match
        video_feat['v0_idx'] = img_index
        video_feat['v1'] = mismatch
        video_feat['v1_idx'] = mismatch_start_index

        seg_start = int(match_start_img_index * self.sr_times)
        eeg_signal = self.all_eeg_signal[person_id][f'{self.seg_id}'][:, seg_start:seg_start + self.window_length]
        img_index = torch.tensor(img_index, dtype=torch.int64, device=self.device)
        person_id = torch.tensor(person_id, dtype=torch.int, device=self.device)
        person_idx = torch.tensor(person_idx, device=self.device)
        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float32, device=self.device)
        eeg_signal = eeg_signal / eeg_signal.abs().max() * 0.8
        eeg_signal = torch.nn.functional.pad(eeg_signal, (0, self.window_length - eeg_signal.shape[1]))
        item = {
            "img_idx": img_index,  # int
            "person_id": person_id,  # int real
            "person_nnid": person_idx,  # int re-indexed
            "eeg": eeg_signal,  # [64,400]
            **video_feat
        }
        return item

    def get_seg_num(self, ):
        L = self.end_img_index - self.start_img_index
        s = self.shift_img_length
        l = self.seg_img_length
        p = self.sep_img_length
        n = int((L - 3 * l - 2 * p) / s)
        self.seg_num = n
        self.left_start_seg_idx = 0

    def __len__(self):
        return self.seg_num * self.person_num


class DilationModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU()):
        super(DilationModel, self).__init__()

        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)

        # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        self.env_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            speech_input_channel = env_input_dimension if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))
            # Dilation on envelope data, share weights
            env_proj_layer = nn.Conv1d(speech_input_channel, dilation_filters,
                                       kernel_size=kernel_size, dilation=1,
                                       stride=1, padding=kernel_size // 2)
            self.env_dilation_layers.append(env_proj_layer)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1, env2):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

            # Dilation on envelope data, share weights
            env_proj_1 = self.env_dilation_layers[layer_index](env_proj_1)
            env_proj_1 = self.activation[layer_index](env_proj_1)
            env_proj_2 = self.env_dilation_layers[layer_index](env_proj_2)
            env_proj_2 = self.activation[layer_index](env_proj_2)

        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)

        return out


class DilationGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU()):
        super(DilationGRUModel, self).__init__()

        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(dilation_filters, dilation_filters, num_layers=1,
                          batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        self.env_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            speech_input_channel = env_input_dimension if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))
            # Dilation on envelope data, share weights
            env_proj_layer = nn.Conv1d(speech_input_channel, dilation_filters,
                                       kernel_size=kernel_size, dilation=1,
                                       stride=1, padding=kernel_size // 2)
            self.env_dilation_layers.append(env_proj_layer)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1, env2):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

            # Dilation on envelope data, share weights
            env_proj_1 = self.env_dilation_layers[layer_index](env_proj_1)
            env_proj_1 = self.activation[layer_index](env_proj_1)
            env_proj_2 = self.env_dilation_layers[layer_index](env_proj_2)
            env_proj_2 = self.activation[layer_index](env_proj_2)
        env_proj_1, _ = self.gru(env_proj_1.permute(0, 2, 1))
        env_proj_1 = env_proj_1.permute(0, 2, 1)
        env_proj_2, _ = self.gru(env_proj_2.permute(0, 2, 1))
        env_proj_2 = env_proj_2.permute(0, 2, 1)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)

        return out


class DilationVideoGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU(), gru_layers=1):
        super(DilationVideoGRUModel, self).__init__()
        #
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(768, dilation_filters, num_layers=gru_layers, batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1=None, env2=None):
        if env1 is not None and env2 is not None:
            env_proj_1 = env1
            env_proj_2 = env2

            # Spatial convolution
            eeg_proj_1 = self.eeg_proj_1(eeg)
            # Construct dilation layers
            for layer_index in range(self.layers):
                # dilation on EEG
                eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
                eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

            env_proj_1, _ = self.gru(env_proj_1.permute(0, 2, 1))
            env_proj_1 = env_proj_1.permute(0, 2, 1)
            env_proj_2, _ = self.gru(env_proj_2.permute(0, 2, 1))
            env_proj_2 = env_proj_2.permute(0, 2, 1)
            # Comparison
            cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
            cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
            # Classification
            out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
            out = self.sigmoid(out).squeeze(-1)

            return out
        else:
            # Spatial convolution
            eeg_proj_1 = self.eeg_proj_1(eeg)
            # Construct dilation layers
            for layer_index in range(self.layers):
                # dilation on EEG
                eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
                eeg_proj_1 = self.activation[layer_index](eeg_proj_1)
            return eeg_proj_1


class DilationVideoGRUCosFeatureModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU(), gru_layers=1):
        super(DilationVideoGRUCosFeatureModel, self).__init__()
        #
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(768, dilation_filters, num_layers=gru_layers, batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=-1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=-1)

        # Classification
        self.fc_layer = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1=None, env2=None):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

        env_proj_1, _ = self.gru(env_proj_1.permute(0, 2, 1))
        env_proj_1 = env_proj_1.permute(0, 2, 1)
        env_proj_2, _ = self.gru(env_proj_2.permute(0, 2, 1))
        env_proj_2 = env_proj_2.permute(0, 2, 1)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)

        return out


class DilationVideoGRUCosFeatureDropoutModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU(), gru_layers=1):
        super(DilationVideoGRUCosFeatureDropoutModel, self).__init__()
        #
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(768, dilation_filters, num_layers=gru_layers, batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=-1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=-1)

        # Classification
        self.fc_layer = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations
        self.dropout = nn.Dropout(0.2)

    def forward(self, eeg, env1=None, env2=None):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

        env_proj_1, _ = self.gru(env_proj_1.permute(0, 2, 1))
        env_proj_1 = env_proj_1.permute(0, 2, 1)
        env_proj_2, _ = self.gru(env_proj_2.permute(0, 2, 1))
        env_proj_2 = env_proj_2.permute(0, 2, 1)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(self.dropout(torch.cat((cos1, cos2), dim=-1)))
        out = self.sigmoid(out).squeeze(-1)

        return out


class OneWayDilationVideoGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU(), gru_layers=1):
        super(OneWayDilationVideoGRUModel, self).__init__()
        #
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(768, dilation_filters, num_layers=gru_layers, batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def get_eeg_emb(self, eeg):
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)
        return eeg_proj_1

    def get_video_emb(self, video):
        env_proj_1, _ = self.gru(video.permute(0, 2, 1))
        env_proj_1 = env_proj_1.permute(0, 2, 1)
        return env_proj_1

    def forward(self, eeg, env1=None, env2=None, return_eeg_proj=True):
        env_proj_1 = self.get_video_emb(env1)
        eeg_proj_1 = self.get_eeg_emb(eeg)
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        out = self.fc_layer(cos1)
        out = self.sigmoid(out).squeeze(-1)
        if not self.training:
            env_proj_2 = self.get_video_emb(env2)
            cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
            out1 = self.fc_layer(cos2)
            out1 = self.sigmoid(out1).squeeze(-1)
            out = (out > out1).float()
        return out


class DilationVideoLSTMModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU(), lstm_layers=1):
        super(DilationVideoLSTMModel, self).__init__()
        #
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.lstm = nn.LSTM(768, dilation_filters, num_layers=lstm_layers,
                            batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1, env2):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

        env_proj_1, _ = self.lstm(env_proj_1.permute(0, 2, 1))
        env_proj_1 = env_proj_1.permute(0, 2, 1)
        env_proj_2, _ = self.lstm(env_proj_2.permute(0, 2, 1))
        env_proj_2 = env_proj_2.permute(0, 2, 1)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)

        return out


class DilationGRUEEGModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5, spatial_filters=8,
                 dilation_filters=16, activation=nn.ReLU()):
        super(DilationGRUEEGModel, self).__init__()

        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)
        self.gru = nn.GRU(dilation_filters, dilation_filters, num_layers=1,
                          batch_first=True)  # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        self.env_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            speech_input_channel = env_input_dimension if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))
            # Dilation on envelope data, share weights
            env_proj_layer = nn.Conv1d(speech_input_channel, dilation_filters,
                                       kernel_size=kernel_size, dilation=1,
                                       stride=1, padding=kernel_size // 2)
            self.env_dilation_layers.append(env_proj_layer)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1, env2):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

            # Dilation on envelope data, share weights
            env_proj_1 = self.env_dilation_layers[layer_index](env_proj_1)
            env_proj_1 = self.activation[layer_index](env_proj_1)
            env_proj_2 = self.env_dilation_layers[layer_index](env_proj_2)
            env_proj_2 = self.activation[layer_index](env_proj_2)
        eeg_proj_1, _ = self.gru(eeg_proj_1.permute(0, 2, 1))
        eeg_proj_1 = eeg_proj_1.permute(0, 2, 1)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1.permute(0, 2, 1), env_proj_1.permute(0, 2, 1))
        cos2 = self.cos_layer_2(eeg_proj_1.permute(0, 2, 1), env_proj_2.permute(0, 2, 1))
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)

        return out


class DilationTransformerModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, kernel_size=5,
                 spatial_filters=8, dilation_filters=16, activation=nn.ReLU(), ):
        super(DilationTransformerModel, self).__init__()
        self.transformer_modeleeg = vit.Transformer(dim=256, depth=3, heads=8, dim_head=16, mlp_dim=256, dropout=0.2)
        self.transformer_modelvideo = vit.Transformer(dim=256, depth=3, heads=8, dim_head=16, mlp_dim=256, dropout=0.2)

        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=1)

        # Construct dilation layers
        self.eeg_dilation_layers = nn.ModuleList()
        self.env_dilation_layers = nn.ModuleList()
        stride_list = [2, 4, 5]
        for layer_index in range(layers):
            # dilation on EEG
            eeg_input_channel = spatial_filters if layer_index == 0 else dilation_filters
            speech_input_channel = env_input_dimension if layer_index == 0 else dilation_filters
            dilated_kernel_size = (kernel_size - 1) * kernel_size ** layer_index + 1
            padding = dilated_kernel_size // 2
            self.eeg_dilation_layers.append(
                nn.Conv1d(eeg_input_channel, dilation_filters,
                          kernel_size=kernel_size, dilation=kernel_size ** layer_index,
                          stride=stride_list[layer_index], padding=padding))
            # Dilation on envelope data, share weights
            env_proj_layer = nn.Conv1d(speech_input_channel, dilation_filters,
                                       kernel_size=kernel_size, dilation=1,
                                       stride=1, padding=kernel_size // 2)
            self.env_dilation_layers.append(env_proj_layer)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(dilation_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

        activations = [activation] * layers
        self.activation = activations

    def forward(self, eeg, env1, env2):
        env_proj_1 = env1
        env_proj_2 = env2

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        # Construct dilation layers
        for layer_index in range(self.layers):
            # dilation on EEG
            eeg_proj_1 = self.eeg_dilation_layers[layer_index](eeg_proj_1)
            eeg_proj_1 = self.activation[layer_index](eeg_proj_1)

            # Dilation on envelope data, share weights
            env_proj_1 = self.env_dilation_layers[layer_index](env_proj_1)
            env_proj_1 = self.activation[layer_index](env_proj_1)
            env_proj_2 = self.env_dilation_layers[layer_index](env_proj_2)
            env_proj_2 = self.activation[layer_index](env_proj_2)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        env_proj_1 = self.transformer_modelvideo(env_proj_1)
        env_proj_2 = self.transformer_modelvideo(env_proj_2)
        # Comparison
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNTransformerModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNTransformerModel, self).__init__()

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=768, depth=layers, heads=8, dim_head=128,
                                                                mlp_dim=768, dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=8, dim_head=128, mlp_dim=768,
                                                        dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNGRUTransformerModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNGRUTransformerModel, self).__init__()
        # 只有EEG一路有参数

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(768, 768, num_layers=1, batch_first=True)
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=768, depth=layers, heads=8, dim_head=128,
                                                                mlp_dim=768, dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=8, dim_head=128, mlp_dim=768,
                                                        dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        eeg_proj_1, _ = self.gru(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNTransformerVideoGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNTransformerVideoGRUModel, self).__init__()
        # EEG一路有参数

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(768, 768, num_layers=1, batch_first=True)
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=768, depth=layers, heads=8, dim_head=128,
                                                                mlp_dim=768, dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=8, dim_head=128, mlp_dim=768,
                                                        dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        env_proj_1, _ = self.gru(env_proj_1)
        env_proj_2, _ = self.gru(env_proj_2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNTransformerM05VideoGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNTransformerM05VideoGRUModel, self).__init__()
        # 和CNNTransformerVideoGRUModel不同的是，transformer出来的特征减去0.5
        # EEG一路有参数

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(768, 768, num_layers=1, batch_first=True)
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=768, depth=layers, heads=8, dim_head=128,
                                                                mlp_dim=768, dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=8, dim_head=128, mlp_dim=768,
                                                        dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1) - 0.5
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        env_proj_1, _ = self.gru(env_proj_1)
        env_proj_2, _ = self.gru(env_proj_2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNTransformer1HeadVideoGRUModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNTransformer1HeadVideoGRUModel, self).__init__()

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(768, 768, num_layers=1, batch_first=True)
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=768, depth=layers, heads=1, dim_head=512,
                                                                mlp_dim=768, dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=1, dim_head=128, mlp_dim=768,
                                                        dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        env_proj_1, _ = self.gru(env_proj_1)
        env_proj_2, _ = self.gru(env_proj_2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class CNNTransformer1HeadVideoGRULowDimModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=768, layers=3, spatial_filters=8, time_emb=False,
                 return_eeg_proj=False):
        super(CNNTransformer1HeadVideoGRULowDimModel, self).__init__()
        # EEG接CNN和transformer，video接gru。CNN将维度变到256，gru也是。

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(768, spatial_filters, num_layers=1, batch_first=True)
        if time_emb:
            self.transformer_modeleeg = TimeEmbeddedTransformer(dim=spatial_filters, depth=layers, heads=1,
                                                                dim_head=spatial_filters, mlp_dim=spatial_filters,
                                                                dropout=0.2)
        else:
            self.transformer_modeleeg = vit.Transformer(dim=spatial_filters, depth=layers, heads=1,
                                                        dim_head=spatial_filters, mlp_dim=spatial_filters, dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)
        self.cos_layer_2 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        env_proj_2 = env_proj_2.swapdims(1, 2)
        env_proj_1, _ = self.gru(env_proj_1)
        env_proj_2, _ = self.gru(env_proj_2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        cos2 = self.cos_layer_2(eeg_proj_1, env_proj_2)
        # Classification
        out = self.fc_layer(torch.cat((cos1, cos2), dim=-1))
        out = self.sigmoid(out).squeeze(-1)
        return out


class OneWayCNNTransformerModel(nn.Module):
    def __init__(self, eeg_input_dimension=64, env_input_dimension=1, layers=3, spatial_filters=8,
                 return_eeg_proj=False):
        super(OneWayCNNTransformerModel, self).__init__()
        # 只要一路视频

        # self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = spatial_filters

        # Spatial convolution
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters, kernel_size=40, stride=40)
        self.relu = nn.ReLU()
        self.transformer_modeleeg = vit.Transformer(dim=768, depth=layers, heads=8, dim_head=128, mlp_dim=768,
                                                    dropout=0.2)

        # Comparison
        self.cos_layer_1 = nn.CosineSimilarity(dim=1)

        # Classification
        self.fc_layer = nn.Linear(spatial_filters, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_eeg_proj = return_eeg_proj

    def forward(self, eeg, env_proj_1=None, env_proj_2=None):

        # Spatial convolution
        eeg_proj_1 = self.eeg_proj_1(eeg)
        eeg_proj_1 = self.relu(eeg_proj_1)
        eeg_proj_1 = eeg_proj_1.swapdims(1, 2)
        eeg_proj_1 = self.transformer_modeleeg(eeg_proj_1)
        if self.return_eeg_proj:
            return eeg_proj_1.swapdims(1, 2)
        # Comparison
        env_proj_1 = env_proj_1.swapdims(1, 2)
        cos1 = self.cos_layer_1(eeg_proj_1, env_proj_1)
        # Classification
        out = self.fc_layer(cos1)
        out = self.sigmoid(out).squeeze(-1)
        if not self.training:  # 在非训练状态下，需要比较两路给的结果，得到结果。
            env_proj_2 = env_proj_2.swapdims(1, 2)
            cos2 = self.cos_layer_1(eeg_proj_1, env_proj_2)
            # Classification
            out2 = self.fc_layer(cos2)
            out2 = self.sigmoid(out2).squeeze(-1)
            out = (out > out2).float()
        return out


def emb_encoding(d_model, max_len, ):
    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(max_len, d_model)
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, max_len)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    # compute positional encoding to consider positional information of words
    return encoding  # [max_len,d_model]


class TimeEmbeddedTransformer(nn.Module):
    def __init__(self, dim=768, depth=3, heads=8, dim_head=128, mlp_dim=768, dropout=0.2, max_len=1000):
        super(TimeEmbeddedTransformer, self).__init__()
        self.transformer = vit.Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                           dropout=dropout)
        self.register_buffer('emb', emb_encoding(dim, max_len))

    def forward(self, x):
        N, L, E = x.shape
        x = x + self.emb[:L]
        return x


def init_models(name):
    if name == 'DilationModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationGRUEEGModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationGRUEEGModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationVideoGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationVideoGRUCosFeatureModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoGRUCosFeatureModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationVideoGRUCosFeatureDropoutModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoGRUCosFeatureDropoutModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'OneWayDilationVideoGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = OneWayDilationVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationVideoGRU2Model':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
            gru_layers=2
        )
    if name == 'DilationVideoGRU3Model':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
            gru_layers=3
        )
    if name == 'DilationVideoLSTMModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoLSTMModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'DilationVideoLSTM2Model':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoLSTMModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
            lstm_layers=2
        )
    if name == 'DilationVideoLSTM3Model':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoLSTMModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
            lstm_layers=3
        )
    if name == 'DilationVideoLSTM4Model':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationVideoLSTMModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
            lstm_layers=4
        )
    if name == 'DilationTransformerModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        layers = 3
        kernel_size = 5
        spatial_filters = 64
        dilation_filters = 256
        activation = nn.ReLU()
        match_model = DilationTransformerModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            kernel_size=kernel_size,
            spatial_filters=spatial_filters,
            dilation_filters=dilation_filters,
            activation=activation,
        )
    if name == 'CNNTransformerModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNTransformerModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )
    if name == 'CNNGRUTransformerModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNGRUTransformerModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )
    if name == 'CNNTransformerVideoGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNTransformerVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )
    if name == 'CNNTransformer1HeadVideoGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNTransformer1HeadVideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )

    if name == 'CNNTransformer1HeadVideoGRULowDimModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 256
        layers = 3
        match_model = CNNTransformer1HeadVideoGRULowDimModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )
    if name == 'CNNTransformerM05VideoGRUModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNTransformerM05VideoGRUModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=False
        )
    if name == 'CNNTimeEmbTransformerModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = CNNTransformerModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
            time_emb=True
        )
    if name == 'OneWayCNNTransformerModel':
        eeg_input_dimension = 64
        env_input_dimension = 768
        spatial_filters = 768
        layers = 3
        match_model = OneWayCNNTransformerModel(
            eeg_input_dimension=eeg_input_dimension,
            env_input_dimension=env_input_dimension,
            layers=layers,
            spatial_filters=spatial_filters,
        )
    return match_model


def load_exp_model(exp_name):
    model_name = exp_name.split('_')[0]
    match_model = init_models(model_name)
    cp_path = f'video_match/{exp_name}/model_best.pth'
    checkpoint = torch.load(cp_path)
    match_model.load_state_dict(checkpoint)
    return match_model


def accuracy(preds, labels):
    """
    计算模型的正确率.

    Args:
        preds (torch.Tensor): 模型的预测结果，形状为 (batch_size, ...)，类型为 torch.float32 或 torch.float64
        labels (torch.Tensor): 样本的实际标签，形状为 (batch_size, ...)，类型为 torch.float32 或 torch.float64

    Returns:
        float: 预测正确的样本数占总样本数的比例
    """
    # 将预测结果转换为二分类的预测标签，即将概率大于 0.5 的样本预测为正样本，否则预测为负样本
    preds = (preds > 0.5).float()

    # 计算预测正确的样本数
    correct = (preds == labels).sum().item()

    # 计算正确率
    accuracy = correct / len(labels)

    return accuracy


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def split_size(length, split_ratio_list):
    size_list = [int(r * length) for r in split_ratio_list]
    size_list[-1] = length - sum(size_list[:-1])
    return size_list


def col_fn(batch):
    batch = torch.utils.data._utils.collate.default_collate(batch)
    keys = list(batch.keys())
    bs = len(batch[keys[0]])
    idx = np.arange(2 * bs).tolist()
    np.random.shuffle(idx)
    new_batch = {}
    new_batch['v0'] = torch.cat([batch['v0'], batch['v1']], dim=0)
    new_batch['v1'] = torch.cat([batch['v1'], batch['v0']], dim=0)
    new_batch['v0_idx'] = torch.cat([batch['v0_idx'], batch['v1_idx']], dim=0)
    new_batch['v1_idx'] = torch.cat([batch['v1_idx'], batch['v0_idx']], dim=0)
    new_batch['match_label'] = torch.cat([torch.ones([bs, ]), torch.zeros([bs, ])], dim=0)
    for key in keys:
        if not key in ['v0', 'v1', 'v0_idx', 'v1_idx']:
            new_batch[key] = torch.cat([batch[key], batch[key]], dim=0)
    for key in new_batch.keys():
        new_batch[key] = new_batch[key][idx]
    return new_batch


def get_data_different_sep_seconds(dataset_class, video, eeg, persons, seg_id, window_seconds, shift_seconds,
                                   sep_seconds, **kwarg):
    dataset = torch.utils.data.ConcatDataset([dataset_class(video,
                                                            eeg,
                                                            persons,
                                                            seg_id=seg_id,
                                                            window_seconds=window_seconds,
                                                            sep_seconds=sep_seconds[i], **kwarg) for i in
                                              range(len(sep_seconds))])
    return dataset


def get_data_different_time(dataset_class, video, eeg, persons, seg_id, window_seconds, shift_seconds, sep_seconds,
                            se_img_index):
    dataset = torch.utils.data.ConcatDataset([get_data_different_sep_seconds(dataset_class, video, eeg, persons, seg_id,
                                                                             window_seconds, shift_seconds, sep_seconds
                                                                             , start_img_index=se_img_index[i][0],
                                                                             end_img_index=se_img_index[i][1]
                                                                             ) for i in range(len(se_img_index))])
    return dataset


def count_trainable_param(model, ):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    num = num / 1e6
    return num


def get_eeg_features(data_loader, model, device):
    eeg_feature_list = []
    video_feature_list = []
    img_idx_list = []
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    with tqdm(data_loader) as pbar:
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg']
            video = batch['v0']
            img_idx = batch['img_idx']
            img_idx_list.append(img_idx)
            eeg = eeg.to(device)
            video = video.to(device)
            img_idx = img_idx.to(device)
            eeg_feature = model.get_eeg_emb(eeg)
            video_feature = model.get_video_emb(video)
            eeg_feature_list.append(eeg_feature.detach().cpu())
            video_feature_list.append(video_feature.detach().cpu())
    eeg_feature_list = torch.cat(eeg_feature_list, 0)
    video_feature_list = torch.cat(video_feature_list, 0)
    img_idx_list = torch.cat(img_idx_list, 0)

    return eeg_feature_list, video_feature_list, img_idx_list


def channel_gradients(model, item):
    eeg = item['eeg'].unsqueeze(0)
    match_video = item['v0']
    match_label = torch.tensor([1], dtype=torch.float32)
    eeg = Variable(eeg, requires_grad=True)
    model.train()
    output = model(eeg, match_video.unsqueeze(0))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, match_label)
    model.zero_grad()
    with torch.no_grad():
        loss.backward()
        gradients = eeg.grad.data.abs().mean(dim=[-1]).squeeze()
    return gradients




def match_video_with_nearest_EEG(eeg, video, model):
    cos_mat = F.cosine_similarity(eeg.unsqueeze(1), video.unsqueeze(0), dim=-1)
    el, vl, cl = cos_mat.shape
    cos_mat = F.sigmoid(
        model.fc_layer(cos_mat.reshape([el * vl, cl]).to(model.fc_layer.weight.data.device))).detach().cpu()
    cos_mat = cos_mat.reshape([el, vl])
    return cos_mat


def evaluate_matching_accuracy(eeg, video, model, k_list=None):
    # random_index=np.arange(len(video))
    # np.random.shuffle(random_index)
    # random_video=video[random_index.tolist()]
    pred_mat = match_video_with_nearest_EEG(eeg, video, model=model).detach().cpu().numpy()
    label = np.tile(np.arange(len(video)), eeg.shape[0] // len(video))
    acc_list = []
    for k in k_list:
        acc = top_k_accuracy_score(label, pred_mat, k=k)
        acc_list.append(acc)
    return acc_list


def allen_frame_id_decode(train_fs, train_labels, test_fs, test_labels, FACTOR=1, allow_arr=3, decoder='knn'):
    time_window = 1

    def feature_for_one_frame(feature):
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        return feature.reshape(-1, FACTOR, feature.shape[-1]).mean(axis=1)

    train_fs = feature_for_one_frame(train_fs)
    test_fs = feature_for_one_frame(test_fs)

    if train_fs is None or test_fs is None:
        return [None], [None], None
    if decoder == 'knn':
        params = np.array([1, 10, 20, 44])
    elif decoder == 'bayes':
        params = np.logspace(-9, 3, 5)
    else:
        raise ValueError('Choose decoder between knn or bayes')
    errs = []

    for n in params:
        if decoder == 'knn':
            train_decoder = KNeighborsClassifier(n_neighbors=n,
                                                 metric='cosine')
        elif decoder == 'bayes':
            train_decoder = GaussianNB(var_smoothing=n)
        train_valid_idx = int(len(train_fs) / 9 * 8)
        train_decoder.fit(train_fs[:train_valid_idx], train_labels[:train_valid_idx])
        pred = train_decoder.predict(train_fs[train_valid_idx:])
        err = train_labels[train_valid_idx:] - pred
        errs.append(abs(err).sum())

    if decoder == 'knn':
        print('k：', params[np.argmin(errs)])
        test_decoder = KNeighborsClassifier(n_neighbors=params[np.argmin(errs)],
                                            metric='cosine')
    elif decoder == 'bayes':
        test_decoder = GaussianNB(var_smoothing=params[np.argmin(errs)])

    test_decoder.fit(train_fs, train_labels)
    pred = test_decoder.predict(test_fs)
    frame_errors = pred - test_labels

    def _quantize_acc(frame_diff, time_window=1):

        true = (abs(frame_diff) <= (time_window * allow_arr)).sum()

        return true / len(frame_diff) * 100

    quantized_acc = _quantize_acc(frame_errors, time_window)

    return pred, frame_errors, quantized_acc


def decoding_video(train_eeg_feature_list, train_img_idx_list,
                   test_eeg_feature_list, test_img_idx_list, func, allow_arr):
    train_eeg_feature_list = func(train_eeg_feature_list)
    test_eeg_feature_list = func(test_eeg_feature_list)

    pred, frame_errors, quantized_acc = allen_frame_id_decode(train_eeg_feature_list.numpy(),
                                                              train_img_idx_list.numpy(),
                                                              test_eeg_feature_list.numpy(), test_img_idx_list.numpy(),
                                                              allow_arr=allow_arr)
    return pred, frame_errors, quantized_acc


def random_guessing(test_img_idx_list, allow_arr):
    length = len(test_img_idx_list)
    labels = np.unique(test_img_idx_list)
    pred = np.random.choice(labels, length)
    frame_errors = pred - test_img_idx_list

    def _quantize_acc(frame_diff, time_window=1):
        true = (abs(frame_diff) <= (time_window * allow_arr)).sum()

        return true / len(frame_diff) * 100

    quantized_acc = _quantize_acc(frame_errors, time_window=1)
    return quantized_acc


def plot_silhouette_score(test_eeg_feature_cluster, test_person_labels, plt_name):
    X = TSNE(n_components=2).fit_transform(test_eeg_feature_cluster)
    sample_silhouette_values = silhouette_samples(test_eeg_feature_cluster, test_person_labels, metric='cosine')

    import matplotlib.cm as cm
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    fig.set_dpi(300)
    y_lower = 10
    n_clusters = len(np.unique(test_person_labels))
    cluster_labels = test_person_labels
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xlim([-1, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    cax = ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    ax2.set_title("The TSNE plot for the various clusters.")
    plt.savefig(mkdir(plt_name))
    plt.show()

