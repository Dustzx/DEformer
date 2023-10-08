import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler
# from conf import device_num
from conf import *
import itertools
import random

device = torch.device('cuda:{}'.format(device_num) if torch.cuda.is_available() else 'cpu')


class SymptomVocab:

    def __init__(self, samples: list = None, add_special_sxs: bool = False,
                 min_sx_freq: int = None, max_voc_size: int = None):

        # sx is short for symptom
        self.sx2idx = {}  # map from symptom to symptom id
        self.idx2sx = {}  # map from symptom id to symptom
        self.sx2count = {}  # map from symptom to symptom count
        self.num_sxs = 0  # number of symptoms

        # symptom attrs
        self.SX_ATTR_PAD_IDX = 0  # symptom attribute id for PAD
        self.SX_ATTR_POS_IDX = 1  # symptom attribute id for YES
        self.SX_ATTR_NEG_IDX = 2  # symptom attribute id for NO
        self.SX_ATTR_NS_IDX = 3  # symptom attribute id for NOT SURE
        self.SX_ATTR_NM_IDX = 4  # symptom attribute id for NOT MENTIONED

        self.SX_ATTR_MAP = {  # map from symptom attribute to symptom attribute id
            '0': self.SX_ATTR_NEG_IDX,
            '1': self.SX_ATTR_POS_IDX,
            '2': self.SX_ATTR_NS_IDX,
        }

        self.SX_ATTR_MAP_INV = {
            self.SX_ATTR_NEG_IDX: '0',
            self.SX_ATTR_POS_IDX: '1',
            self.SX_ATTR_NS_IDX: '2',
        }

        # special symptoms
        self.num_special = 0  # number of special symptoms
        self.special_sxs = []

        # vocabulary
        self.min_sx_freq = min_sx_freq  # minimal symptom frequency
        self.max_voc_size = max_voc_size  # maximal symptom size

        # add special symptoms
        if add_special_sxs:  # special symptoms
            self.SX_PAD = '[PAD]'
            self.SX_START = '[START]'
            self.SX_END = '[END]'
            self.SX_UNK = '[UNKNOWN]'
            self.SX_CLS = '[CLS]'
            self.SX_MASK = '[MASK]'
            self.special_sxs.extend([self.SX_PAD, self.SX_START, self.SX_END, self.SX_UNK, self.SX_CLS, self.SX_MASK])
            self.sx2idx = {sx: idx for idx, sx in enumerate(self.special_sxs)}
            self.idx2sx = {idx: sx for idx, sx in enumerate(self.special_sxs)}
            self.num_special = len(self.special_sxs)
            self.num_sxs += self.num_special

        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('symptom vocabulary constructed using {} split and {} samples '
                  '({} symptoms with {} special symptoms)'.
                  format(len(samples), num_samples, self.num_sxs - self.num_special, self.num_special))

        # trim vocabulary
        self.trim_voc()

        assert self.num_sxs == len(self.sx2idx) == len(self.idx2sx)

    def add_symptom(self, sx: str) -> None:
        if sx not in self.sx2idx:
            self.sx2idx[sx] = self.num_sxs
            self.sx2count[sx] = 1
            self.idx2sx[self.num_sxs] = sx
            self.num_sxs += 1
        else:
            self.sx2count[sx] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            for sx in sample['exp_sxs']:
                self.add_symptom(sx)
            for sx in sample['imp_sxs']:
                self.add_symptom(sx)
        return len(samples)

    def trim_voc(self):
        sxs = [sx for sx in sorted(self.sx2count, key=self.sx2count.get, reverse=True)]
        if self.min_sx_freq is not None:
            sxs = [sx for sx in sxs if self.sx2count.get(sx) >= self.min_sx_freq]
        if self.max_voc_size is not None:
            sxs = sxs[: self.max_voc_size]
        sxs = self.special_sxs + sxs
        self.sx2idx = {sx: idx for idx, sx in enumerate(sxs)}
        self.idx2sx = {idx: sx for idx, sx in enumerate(sxs)}
        self.sx2count = {sx: self.sx2count.get(sx) for sx in sxs if sx in self.sx2count}
        self.num_sxs = len(self.sx2idx)
        print('trimmed to {} symptoms with {} special symptoms'.
              format(self.num_sxs - self.num_special, self.num_special))

    def encode(self, sxs: dict, keep_unk=True, add_start=False, add_end=False):
        sx_ids, attr_ids = [], []
        if add_start:
            sx_ids.append(self.start_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        for sx, attr in sxs.items():
            if sx in self.sx2idx:
                sx_ids.append(self.sx2idx.get(sx))
                attr_ids.append(self.SX_ATTR_MAP.get(attr))
            else:
                if keep_unk:
                    sx_ids.append(self.unk_idx)
                    attr_ids.append(self.SX_ATTR_MAP.get(attr))
        if add_end:
            sx_ids.append(self.end_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        return sx_ids, attr_ids

    def decoder(self, sx_ids, attr_ids):
        sx_attr = {}
        for sx_id, attr_id in zip(sx_ids, attr_ids):
            if attr_id not in [self.SX_ATTR_PAD_IDX, self.SX_ATTR_NM_IDX]:
                sx_attr.update({self.idx2sx.get(sx_id): self.SX_ATTR_MAP_INV.get(attr_id)})
        return sx_attr

    def __len__(self) -> int:
        return self.num_sxs

    @property
    def pad_idx(self) -> int:
        return self.sx2idx.get(self.SX_PAD)

    @property
    def start_idx(self) -> int:
        return self.sx2idx.get(self.SX_START)

    @property
    def end_idx(self) -> int:
        return self.sx2idx.get(self.SX_END)

    @property
    def unk_idx(self) -> int:
        return self.sx2idx.get(self.SX_UNK)

    @property
    def cls_idx(self) -> int:
        return self.sx2idx.get(self.SX_CLS)

    @property
    def mask_idx(self) -> int:
        return self.sx2idx.get(self.SX_MASK)

    @property
    def pad_sx(self) -> str:
        return self.SX_PAD

    @property
    def start_sx(self) -> str:
        return self.SX_START

    @property
    def end_sx(self) -> str:
        return self.SX_END

    @property
    def unk_sx(self) -> str:
        return self.SX_UNK

    @property
    def cls_sx(self) -> str:
        return self.SX_CLS

    @property
    def mask_sx(self) -> str:
        return self.SX_MASK


class DiseaseVocab:

    def __init__(self, samples: list = None):

        # dis is short for disease
        self.dis2idx = {}
        self.idx2dis = {}
        self.dis2count = {}
        self.num_dis = 0

        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('disease vocabulary constructed using {} split and {} samples\nnum of unique diseases: {}'.
                  format(len(samples), num_samples, self.num_dis))

    def add_disease(self, dis: str) -> None:
        if dis not in self.dis2idx:
            self.dis2idx[dis] = self.num_dis
            self.dis2count[dis] = 1
            self.idx2dis[self.num_dis] = dis
            self.num_dis += 1
        else:
            self.dis2count[dis] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            self.add_disease(sample['label'])
        return len(samples)

    def __len__(self) -> int:
        return self.num_dis

    def encode(self, dis):
        return self.dis2idx.get(dis)


class SymptomDataset(Dataset):

    def __init__(self, samples, sv: SymptomVocab, dv: DiseaseVocab, keep_unk: bool,
                 add_src_start: bool = False, add_tgt_start: bool = False, add_tgt_end: bool = False):
        self.samples = samples
        self.sv = sv
        self.dv = dv
        self.keep_unk = keep_unk
        self.size = len(self.sv)
        self.add_src_start = add_src_start
        self.add_tgt_start = add_tgt_start
        self.add_tgt_end = add_tgt_end

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        exp_sx_ids, exp_attr_ids = self.sv.encode(
            sample['exp_sxs'], keep_unk=self.keep_unk, add_start=self.add_src_start)
        imp_sx_ids, imp_attr_ids = self.sv.encode(
            sample['imp_sxs'], keep_unk=self.keep_unk, add_start=self.add_tgt_start, add_end=self.add_tgt_end)
        exp_sx_ids, exp_attr_ids, imp_sx_ids, imp_attr_ids, label = to_tensor_vla(
            exp_sx_ids, exp_attr_ids, imp_sx_ids, imp_attr_ids, self.dv.encode(sample['label']), dtype=torch.long)
        item = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'label': label
        }
        return item


# language model
def lm_collater(samples):
    sx_ids = pad_sequence(
        [torch.cat([sample['exp_sx_ids'], sample['imp_sx_ids']]) for sample in samples], padding_value=0)
    attr_ids = pad_sequence(
        [torch.cat([sample['exp_attr_ids'], sample['imp_attr_ids']]) for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    items = {
        'sx_ids': sx_ids,
        'attr_ids': attr_ids,
        'labels': labels
    }
    return items


def lm_collater_shuffle(samples, sscom, median_list, avg_list):
    # global sscom
    # global avg_ss
    # global median_list

    if is_shuffle:
        if full_shuffle:
            # 已知症状直接打乱,注意保持症状与attr的对应关系
            exp_list = [sample['exp_sx_ids'] for sample in samples]
            exp_attr_list = [sample['exp_attr_ids'] for sample in samples]
            # 打包为一个元组列表
            # paired_list = list(zip(exp_list, exp_attr_list))
            for i in range(len(exp_list)):
                idx = torch.randperm(len(exp_list[i]))
                exp_list[i] = exp_list[i][idx]
                exp_attr_list[i] = exp_attr_list[i][idx]
            # 待询问症状直接打乱,注意保持症状与attr的对应关系
            imp_list = [sample['imp_sx_ids'] for sample in samples]
            imp_attr_list = [sample['imp_attr_ids'] for sample in samples]

            for i in range(len(imp_list)):
                idx = torch.randperm(len(imp_list[i]) - 2)
                idx += 1
                idx = torch.cat((torch.tensor([0]), idx, torch.tensor([len(imp_list[i]) - 1])))
                imp_list[i] = imp_list[i][idx]
                imp_attr_list[i] = imp_attr_list[i][idx]
        else:
            # 已知症状直接打乱,注意保持症状与attr的对应关系
            exp_list = [sample['exp_sx_ids'] for sample in samples]
            exp_attr_list = [sample['exp_attr_ids'] for sample in samples]
            # 打包为一个元组列表
            # paired_list = list(zip(exp_list, exp_attr_list))
            for i in range(len(exp_list)):
                idx = torch.randperm(len(exp_list[i]))
                exp_list[i] = exp_list[i][idx]
                exp_attr_list[i] = exp_attr_list[i][idx]

            imp_list = [sample['imp_sx_ids'] for sample in samples]
            imp_attr_list = [sample['imp_attr_ids'] for sample in samples]
            # paired_list = list(zip(imp_list, imp_attr_list))
            # 未知症状先根据疾病症状共现矩阵dscom和症状症状共现矩阵sscom找到症状群后,从症状群内以及症状群级别打乱
            for sub_idx in range(len(imp_list)):
                list = []
                rd_idx = []
                shift_range = 0
                last_sym = None
                for i, sym in enumerate(imp_list[sub_idx]):
                    if last_sym == None:
                        last_sym = sym
                        # list.append(sym)
                        shift_range = i + 1
                        rd_idx.append(i)
                        continue
                    if sscom[sym, last_sym] < avg_list[sym] and i != 1:
                        if len(list) != 0:
                            idx = torch.randperm(len(list))
                            idx += shift_range
                            rd_idx.append(idx)
                        rd_idx.append(i)
                        list = []
                        shift_range = i + 1
                        last_sym = sym
                    else:
                        last_sym = sym
                        list.append(sym)
                # 将rd_idx中的tensor索引顺序进行随机交换
                # 找到所有包含tensor的元素的索引
                origin_idx = [i for i, x in enumerate(rd_idx) if isinstance(x, torch.Tensor)]
                # 随机打乱索引
                shuff_index = origin_idx.copy()
                random.shuffle(shuff_index)
                # 先将rd_idx复制一份，再将打乱后的顺序赋值给rd_idx
                rd_idx_copy = rd_idx.copy()
                for i, idx in enumerate(origin_idx):
                    rd_idx_copy[idx] = rd_idx[shuff_index[i]]
                rd_idx = rd_idx_copy
                # 将rd_idx中的int和tensor数据统一格式为tensor,并合并成一维tensor
                for i in range(len(rd_idx)):
                    if isinstance(rd_idx[i], int):
                        rd_idx[i] = torch.tensor([rd_idx[i]])
                rd_idx = torch.cat(rd_idx)

                imp_list[sub_idx] = imp_list[sub_idx][rd_idx]
                imp_attr_list[sub_idx] = imp_attr_list[sub_idx][rd_idx]

        # 将打乱后的exp_list imp_list进行拼接并padding
        sx_ids = pad_sequence(
            [torch.cat([exp_sub, imp_sub]) for exp_sub, imp_sub in zip(exp_list, imp_list)], padding_value=0)
        attr_ids = pad_sequence(
            [torch.cat([exp_sub, imp_sub]) for exp_sub, imp_sub in zip(exp_attr_list, imp_attr_list)], padding_value=0)
        labels = torch.stack([sample['label'] for sample in samples])
        items = {
            'sx_ids': sx_ids,
            'attr_ids': attr_ids,
            'labels': labels
        }
    else:
        sx_ids = pad_sequence(
            [torch.cat([sample['exp_sx_ids'], sample['imp_sx_ids']]) for sample in samples], padding_value=0)
        attr_ids = pad_sequence(
            [torch.cat([sample['exp_attr_ids'], sample['imp_attr_ids']]) for sample in samples], padding_value=0)
        labels = torch.stack([sample['label'] for sample in samples])
        items = {
            'sx_ids': sx_ids,
            'attr_ids': attr_ids,
            'labels': labels
        }
    return items


def pg_collater_shuffle(samples, sscom, median_list,avg_list):
    # global sscom
    # global avg_ss
    # global median_list

    if is_shuffle:
        if full_shuffle:
            # 已知症状直接打乱,注意保持症状与attr的对应关系
            exp_list = [sample['exp_sx_ids'] for sample in samples]
            exp_attr_list = [sample['exp_attr_ids'] for sample in samples]
            # 打包为一个元组列表
            # paired_list = list(zip(exp_list, exp_attr_list))
            for i in range(len(exp_list)):
                idx = torch.randperm(len(exp_list[i]))
                exp_list[i] = exp_list[i][idx]
                exp_attr_list[i] = exp_attr_list[i][idx]
            # 待询问症状直接打乱,注意保持症状与attr的对应关系
            imp_list = [sample['imp_sx_ids'] for sample in samples]
            imp_attr_list = [sample['imp_attr_ids'] for sample in samples]

            for i in range(len(imp_list)):
                idx = torch.randperm(len(imp_list[i]) - 2)
                idx += 1
                idx = torch.cat((torch.tensor([0]), idx, torch.tensor([len(imp_list[i]) - 1])))
                imp_list[i] = imp_list[i][idx]
                imp_attr_list[i] = imp_attr_list[i][idx]
        else:
            # 已知症状直接打乱,注意保持症状与attr的对应关系
            exp_list = [sample['exp_sx_ids'] for sample in samples]
            exp_attr_list = [sample['exp_attr_ids'] for sample in samples]
            # 打包为一个元组列表
            # paired_list = list(zip(exp_list, exp_attr_list))
            for i in range(len(exp_list)):
                idx = torch.randperm(len(exp_list[i]))
                exp_list[i] = exp_list[i][idx]
                exp_attr_list[i] = exp_attr_list[i][idx]

            imp_list = [sample['imp_sx_ids'] for sample in samples]
            imp_attr_list = [sample['imp_attr_ids'] for sample in samples]
            # paired_list = list(zip(imp_list, imp_attr_list))
            # 未知症状先根据疾病症状共现矩阵dscom和症状症状共现矩阵sscom找到症状群后,从症状群内以及症状群级别打乱
            for sub_idx in range(len(imp_list)):
                list = []
                rd_idx = []
                shift_range = 0
                last_sym = None
                for i, sym in enumerate(imp_list[sub_idx]):
                    if last_sym == None:
                        last_sym = sym
                        # list.append(sym)
                        shift_range = i + 1
                        rd_idx.append(i)
                        continue

                    # 计算当前sym与在list中所有症状共现的平均值
                    avg_ss = sum(sscom[sym, j] for j in list)
                    if avg_ss != 0:
                        avg_ss = avg_ss / len(list)
                    if avg_ss < avg_list[sym] and i != 1:
                        if len(list) != 0:
                            idx = torch.randperm(len(list))
                            idx += shift_range
                            rd_idx.append(idx)
                            # 使用group以二维数组的形式将记录每个打乱后的症状群
                            # sym_group[group_num] = list
                            # group_num += 1
                        rd_idx.append(i)
                        list = []
                        shift_range = i + 1
                        last_sym = sym
                    else:
                        last_sym = sym
                        list.append(sym)
                # 将rd_idx中的tensor索引顺序进行随机交换
                # 找到所有包含tensor的元素的索引
                origin_idx = [i for i, x in enumerate(rd_idx) if isinstance(x, torch.Tensor)]
                # 随机打乱索引
                shuff_index = origin_idx.copy()
                random.shuffle(shuff_index)
                # 先将rd_idx复制一份，再将打乱后的顺序赋值给rd_idx
                rd_idx_copy = rd_idx.copy()
                for i, idx in enumerate(origin_idx):
                    rd_idx_copy[idx] = rd_idx[shuff_index[i]]
                rd_idx = rd_idx_copy

                # 将rd_idx中的int和tensor数据统一格式为tensor,并合并成一维tensor
                for i in range(len(rd_idx)):
                    if isinstance(rd_idx[i], int):
                        rd_idx[i] = torch.tensor([rd_idx[i]])
                rd_idx = torch.cat(rd_idx)

                imp_list[sub_idx] = imp_list[sub_idx][rd_idx]
                imp_attr_list[sub_idx] = imp_attr_list[sub_idx][rd_idx]

        # 将打乱后的exp_list imp_list进行拼接并padding
        exp_sx_ids = pad_sequence([exp for exp in exp_list], padding_value=0)
        exp_attr_ids = pad_sequence([exp_attr for exp_attr in exp_attr_list], padding_value=0)
        imp_sx_ids = pad_sequence([imp for imp in imp_list], padding_value=0)
        imp_attr_ids = pad_sequence([imp_attr for imp_attr in imp_attr_list], padding_value=0)
        labels = torch.stack([sample['label'] for sample in samples])
        items = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'labels': labels
        }
    else:
        exp_sx_ids = pad_sequence([sample['exp_sx_ids'] for sample in samples], padding_value=0)
        exp_attr_ids = pad_sequence([sample['exp_attr_ids'] for sample in samples], padding_value=0)
        imp_sx_ids = pad_sequence([sample['imp_sx_ids'] for sample in samples], padding_value=0)
        imp_attr_ids = pad_sequence([sample['imp_attr_ids'] for sample in samples], padding_value=0)
        labels = torch.stack([sample['label'] for sample in samples])
        items = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'labels': labels
        }
    return items


# policy gradient
def pg_collater(samples):
    exp_sx_ids = pad_sequence([sample['exp_sx_ids'] for sample in samples], padding_value=0)
    exp_attr_ids = pad_sequence([sample['exp_attr_ids'] for sample in samples], padding_value=0)
    imp_sx_ids = pad_sequence([sample['imp_sx_ids'] for sample in samples], padding_value=0)
    imp_attr_ids = pad_sequence([sample['imp_attr_ids'] for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    items = {
        'exp_sx_ids': exp_sx_ids,
        'exp_attr_ids': exp_attr_ids,
        'imp_sx_ids': imp_sx_ids,
        'imp_attr_ids': imp_attr_ids,
        'labels': labels
    }
    return items


class PatientSimulator:

    def __init__(self, sv: SymptomVocab):
        self.sv = sv

    def init_sx_ids(self, bsz):
        return torch.full((1, bsz), self.sv.start_idx, dtype=torch.long, device=device)

    def init_attr_ids(self, bsz):
        return torch.full((1, bsz), self.sv.SX_ATTR_PAD_IDX, dtype=torch.long, device=device)

    def answer(self, action, batch):
        d_action, attr_ids = [], []
        for idx, act in enumerate(action):
            if act.item() < self.sv.num_special:
                attr_ids.append(self.sv.SX_ATTR_PAD_IDX)
                d_action.append(self.sv.pad_idx)
            else:
                indices = batch['imp_sx_ids'][:, idx].eq(act.item()).nonzero(as_tuple=False)
                if len(indices) > 0:
                    attr_ids.append(batch['imp_attr_ids'][indices[0].item(), idx].item())
                    d_action.append(act)
                else:
                    attr_ids.append(self.sv.SX_ATTR_NM_IDX)
                    d_action.append(self.sv.pad_idx)
        return to_tensor_vla(d_action, attr_ids)


# symptom inquiry epoch recorder
class SIRecorder:

    def __init__(self, num_samples, num_imp_sxs, digits):
        self.epoch_rewards = defaultdict(list)  # 模拟的每一个序列的每一个查询的症状的奖励
        self.epoch_num_turns = defaultdict(list)  # 模拟的每一个序列的询问的轮数

        self.epoch_num_hits = defaultdict(list)  # 模拟的每一个序列的症状命中总个数
        self.epoch_num_pos_hits = defaultdict(list)  # 模拟的每一个序列的症状（yes）命中个数
        self.epoch_num_neg_hits = defaultdict(list)  # 模拟的每一个序列的症状（no）命中个数
        self.epoch_num_ns_hits = defaultdict(list)  # 模拟的每一个序列的症状（not sure）命中个数

        self.epoch_num_repeats = defaultdict(list)  # 模拟的每一个序列的询问的重复症状的个数
        self.epoch_distances = defaultdict(list)  # 模拟的每一个序列的杰卡德距离（不考虑顺序）
        self.epoch_bleus = defaultdict(list)  # 模拟的每一个序列的BLEU（考虑顺序）

        self.num_samples = num_samples

        num_pos_imp_sxs, num_neg_imp_sxs, num_ns_imp_sxs = num_imp_sxs
        self.num_pos_imp_sxs = num_pos_imp_sxs
        self.num_neg_imp_sxs = num_neg_imp_sxs
        self.num_ns_imp_sxs = num_ns_imp_sxs
        self.num_imp_sxs = sum(num_imp_sxs)
        self.digits = digits

        self.epoch_acc = defaultdict(float)

    def update(self, batch_rewards, batch_num_turns, batch_num_hits, batch_num_pos_hits,
               batch_num_neg_hits, batch_num_ns_hits, batch_num_repeats, batch_distances, batch_bleus, epoch):
        self.epoch_rewards[epoch].extend(batch_rewards)
        self.epoch_num_turns[epoch].extend(batch_num_turns)

        self.epoch_num_hits[epoch].extend(batch_num_hits)
        self.epoch_num_pos_hits[epoch].extend(batch_num_pos_hits)
        self.epoch_num_neg_hits[epoch].extend(batch_num_neg_hits)
        self.epoch_num_ns_hits[epoch].extend(batch_num_ns_hits)

        self.epoch_num_repeats[epoch].extend(batch_num_repeats)

        self.epoch_distances[epoch].extend(batch_distances)
        self.epoch_bleus[epoch].extend(batch_bleus)

    def update_acc(self, epoch, acc):
        self.epoch_acc[epoch] = acc

    def epoch_summary(self, epoch, logger):
        avg_epoch_rewards = average(self.epoch_rewards[epoch], self.num_imp_sxs)
        avg_epoch_turns = average(self.epoch_num_turns[epoch], self.num_samples)

        avg_epoch_num_hits = average(self.epoch_num_hits[epoch], self.num_imp_sxs)
        avg_epoch_num_pos_hits = average(self.epoch_num_pos_hits[epoch], self.num_pos_imp_sxs)
        avg_epoch_num_neg_hits = average(self.epoch_num_neg_hits[epoch], self.num_neg_imp_sxs)
        avg_epoch_num_ns_hits = average(self.epoch_num_ns_hits[epoch], self.num_ns_imp_sxs)

        avg_epoch_num_repeats = average(self.epoch_num_repeats[epoch], self.num_samples)
        avg_epoch_distances = average(self.epoch_distances[epoch], self.num_samples)
        avg_epoch_bleus = average(self.epoch_bleus[epoch], self.num_samples)

        epoch_acc = self.epoch_acc[epoch] if epoch in self.epoch_acc else 0
        ####
        logger.info(
            'epoch: {} -> rewards: {}, all/pos/neg/ns hits: {}/{}/{}/{}, acc: {}, repeats: {}, turns: {}, distances: {}, bleus: {}'.
            format(epoch + 1,
                   round(avg_epoch_rewards, self.digits),
                   round(avg_epoch_num_hits, self.digits),
                   round(avg_epoch_num_pos_hits, self.digits),
                   round(avg_epoch_num_neg_hits, self.digits),
                   round(avg_epoch_num_ns_hits, self.digits),
                   round(epoch_acc, self.digits),
                   round(avg_epoch_num_repeats, self.digits),
                   round(avg_epoch_turns, self.digits),
                   round(avg_epoch_distances, self.digits),
                   round(avg_epoch_bleus, self.digits)))
        ####
        # print(
        #     'epoch: {} -> rewards: {}, all/pos/neg/ns hits: {}/{}/{}/{}, acc: {}, repeats: {}, turns: {}, distances: {}, bleus: {}'.
        #     format(epoch + 1,
        #            round(avg_epoch_rewards, self.digits),
        #            round(avg_epoch_num_hits, self.digits),
        #            round(avg_epoch_num_pos_hits, self.digits),
        #            round(avg_epoch_num_neg_hits, self.digits),
        #            round(avg_epoch_num_ns_hits, self.digits),
        #            round(epoch_acc, self.digits),
        #            round(avg_epoch_num_repeats, self.digits),
        #            round(avg_epoch_turns, self.digits),
        #            round(avg_epoch_distances, self.digits),
        #            round(avg_epoch_bleus, self.digits)))

    @staticmethod
    def lmax(arrays: list):
        cur_val = arrays[-1]
        max_val = max(arrays)
        max_index = len(arrays) - arrays[::-1].index(max_val) - 1
        return cur_val, max_val, max_index

    def report(self, max_epoch: int, digits: int, alpha: float = 0.2, verbose: bool = False):
        recs = [average(self.epoch_num_hits[epoch], self.num_imp_sxs) for epoch in range(max_epoch + 1)]
        cur_rec, best_rec, best_rec_epoch = self.lmax(recs)
        accs = [self.epoch_acc[epoch] for epoch in range(max_epoch + 1)]
        cur_acc, best_acc, best_acc_epoch = self.lmax(accs)
        mets = [alpha * rec + (1 - alpha) * acc for rec, acc in zip(recs, accs)]  # 0.2*recall+0.8*accuracy
        cur_met, best_met, best_met_epoch = self.lmax(mets)
        best_rec_acc, best_acc_rec, best_met_rec, best_met_acc = \
            accs[best_rec_epoch], recs[best_acc_epoch], recs[best_met_epoch], accs[best_met_epoch]
        if verbose:
            print(
                'best recall -> epoch: {}, recall: {}, accuracy: {}\nbest accuracy -> epoch: {}, recall: {}, accuracy: {}\nbest metric -> epoch: {}, recall: {}, accuracy: {}'.format(
                    best_rec_epoch + 1, round(best_rec, digits), round(best_rec_acc, digits),
                    best_acc_epoch + 1, round(best_acc_rec, digits), round(best_acc, digits),
                    best_met_epoch + 1, round(best_met_rec, digits), round(best_met_acc, digits)
                ))
        return cur_rec, best_rec, best_rec_epoch, best_rec_acc, cur_acc, best_acc, best_acc_epoch, best_acc_rec, cur_met, best_met, best_met_epoch, best_met_rec, best_met_acc


def recursive_sum(item):
    if isinstance(item, list):
        try:
            return sum(item)
        except TypeError:
            return recursive_sum(sum(item, []))
    else:
        return item


def average(numerator, denominator):
    return 0 if recursive_sum(denominator) == 0 else recursive_sum(numerator) / recursive_sum(denominator)


def to_numpy(tensors):
    arrays = {}
    for key, tensor in tensors.items():
        arrays[key] = tensor.cpu().numpy()
    return arrays


def to_numpy_(tensor):
    return tensor.cpu().numpy()


def to_list(tensor):
    return to_numpy_(tensor).tolist()


def to_numpy_vla(*tensors):
    arrays = []
    for tensor in tensors:
        arrays.append(to_numpy_(tensor))
    return arrays


def to_tensor_(array, dtype=None):
    if dtype is None:
        return torch.tensor(array, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)


def to_tensor_vla(*arrays, dtype=None):
    tensors = []
    for array in arrays:
        tensors.append(to_tensor_(array, dtype))
    return tensors


def compute_num_sxs(samples):
    num_yes_imp_sxs = 0
    num_no_imp_sxs = 0
    num_not_sure_imp_sxs = 0
    for sample in samples:
        for sx, attr in sample['imp_sxs'].items():
            if attr == '0':
                num_no_imp_sxs += 1
            elif attr == '1':
                num_yes_imp_sxs += 1
            else:
                num_not_sure_imp_sxs += 1
    return num_yes_imp_sxs, num_no_imp_sxs, num_not_sure_imp_sxs


def combinations(iterable, r):  # 从iterable中取出r个元素的所有组合
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# compute co-occurrence matrix
def compute_dscom(samples, sv: SymptomVocab, dv: DiseaseVocab, smooth: bool = False):
    cm = np.zeros((dv.num_dis, sv.num_sxs))
    if not isinstance(samples, tuple):
        samples = (samples,)
    for split in samples:
        for sample in split:
            exp_sx_ids, _ = sv.encode(sample['exp_sxs'])
            imp_sx_ids, _ = sv.encode(sample['imp_sxs'])
            dis_id = dv.encode(sample['label'])
            for sx_id in exp_sx_ids + imp_sx_ids:
                cm[dis_id][sx_id] += 1
    if smooth:
        for i in range(dv.num_dis):
            row_sum = np.sum(cm[i])
            if row_sum > 0:
                for j in range(sv.num_sxs):
                    if cm[i][j] > 0:
                        cm[i][j] /= row_sum
    return cm


def compute_sscom(samples, sv: SymptomVocab, smooth: bool = False, normalize: bool = True):
    cm = np.zeros((sv.num_sxs, sv.num_sxs))
    row_sums = []
    median_list = []
    average_list = []
    for sample in samples:
        exp_sx_ids, _ = sv.encode(sample['exp_sxs'])
        imp_sx_ids, _ = sv.encode(sample['imp_sxs'])
        sxs = exp_sx_ids + imp_sx_ids
        for (i, j) in combinations(sxs, 2):
            if i != j:
                cm[i][j] += 1
                cm[j][i] += 1
    # 遍历cm计算每行的中位数
    for i in range(sv.num_sxs):
        median_list.append(np.median([num for num in cm[i] if num != 0]))
    # 遍历cm，计算每行的平均数，并存储在average_list中
    for i in range(sv.num_sxs):
        average_list.append(np.mean([num for num in cm[i] if num != 0]))
    if smooth:
        min_val, max_val = .1, .5
        for i in range(sv.num_sxs):
            for j in range(sv.num_sxs):
                if cm[i][j] == 0:
                    cm[i][j] = truncnorm.rvs(min_val, max_val, size=1)[0]  # rvs可以实现截断正态分布，参数分别为截断区间的下限，上限，size
    if normalize:
        # cm = cm / cm.sum(axis=1).reshape(-1, 1)# normalize作用是将每一行的和变为1

        for i in range(sv.num_sxs):
            row_sum = np.sum(cm[i])
            row_sums.append(row_sum)
            if row_sum > 0:
                for j in range(sv.num_sxs):
                    if cm[i][j] > 0:
                        cm[i][j] /= row_sum
    return cm, median_list, average_list


# def compute_sscom(samples, sv: SymptomVocab, smooth: bool = False, normalize: bool = True):
#     num_sxs = sv.num_sxs
#     cm = np.zeros((num_sxs, num_sxs))
#     for sample in samples:
#         exp_sx_ids, _ = sv.encode(sample['exp_sxs'])
#         imp_sx_ids, _ = sv.encode(sample['imp_sxs'])
#         sxs = exp_sx_ids + imp_sx_ids
#         for (i, j) in combinations(sxs, 2):
#             cm[i][j] += 1
#     if smooth:
#         min_val, max_val = .1, .5
#         for i in range(num_sxs):
#             for j in range(i+1, num_sxs):
#                 if cm[i][j] == 0:
#                     cm[i][j] = truncnorm.rvs(min_val, max_val, size=1)[0]
#                     cm[j][i] = cm[i][j]
#     if normalize:
#         for i in range(num_sxs):
#             row_sum = np.sum(cm[i])
#             if row_sum > 0:
#                 cm[i] /= row_sum
#     return cm

def random_agent(samples, sv: SymptomVocab, max_turn: int, exclude_exp: bool = True, times: int = 100):
    recs = []
    for _ in range(times):
        num_imp_sxs, num_hits = 0, 0
        for sample in samples:
            exp_sx_ids, _ = sv.encode(sample['exp_sxs'], keep_unk=False)
            imp_sx_ids, _ = sv.encode(sample['imp_sxs'], keep_unk=False)
            if exclude_exp:
                action_space = [sx_id for sx_id, _ in sv.idx2sx.items() if sx_id not in exp_sx_ids]
            else:
                action_space = [sx_id for sx_id, _ in sv.idx2sx.items()]
            actions = np.random.choice(action_space, size=max_turn, replace=False)
            num_imp_sxs += len(imp_sx_ids)
            num_hits += len([action for action in actions if action in imp_sx_ids])
        recs.append(num_hits / num_imp_sxs)
    return recs


def rule_agent(samples, cm, sv: SymptomVocab, max_turn: int, exclude_exp: bool = True):
    num_imp_sxs, num_pos_imp_sxs, num_neg_imp_sxs = 0, 0, 0
    num_hits, num_pos_hits, num_neg_hits = 0, 0, 0
    for sample in samples:
        exp_sx_ids, _ = sv.encode(sample['exp_sxs'], keep_unk=False)
        imp_sx_ids, imp_attr_ids = sv.encode(sample['imp_sxs'], keep_unk=False)
        imp_pos_sx_ids = [sx_id for sx_id, attr_id in zip(imp_sx_ids, imp_attr_ids) if attr_id == sv.SX_ATTR_POS_IDX]
        imp_neg_sx_ids = [sx_id for sx_id, attr_id in zip(imp_sx_ids, imp_attr_ids) if attr_id == sv.SX_ATTR_NEG_IDX]
        num_imp_sxs += len(imp_sx_ids)
        num_pos_imp_sxs += len(imp_pos_sx_ids)
        num_neg_imp_sxs += len(imp_neg_sx_ids)
        actions = []
        current = set(exp_sx_ids)
        previous = set()
        for step in range(max_turn):
            # similarity score
            sim = np.zeros(sv.num_sxs)
            for sx in current:
                sim += cm[sx]
            index = -1
            if exclude_exp:
                for index in np.flip(np.argsort(sim)):
                    if index not in current.union(previous):
                        break
            else:
                for index in np.flip(np.argsort(sim)):
                    if index not in previous:
                        break
            # if index in imp_sx_ids and imp_attr_ids[imp_sx_ids.index(index)] == sv.SX_ATTR_POS_IDX:
            #     current.add(index)
            # if index in imp_sx_ids:
            #     current.add(index)
            previous.add(index)
            actions.append(index)
        num_hits += len([sx_id for sx_id in actions if sx_id in imp_sx_ids])
        num_pos_hits += len([sx_id for sx_id in actions if sx_id in imp_pos_sx_ids])
        num_neg_hits += len([sx_id for sx_id in actions if sx_id in imp_neg_sx_ids])
    rec = num_hits / num_imp_sxs
    pos_rec = num_pos_hits / num_pos_imp_sxs
    neg_rec = num_neg_hits / num_neg_imp_sxs
    return rec, pos_rec, neg_rec


def count_permutations_overlap(seq1, seq2):
    """
    计算两个序列内多个元素的所有形式排列组合的重合次数
    """
    # 将序列转换为集合
    set1 = set(seq1)
    set2 = set(seq2)

    # 计算交集
    set3 = set1 & set2

    # 遍历交集中的每个元素，计算出现次数并累加
    overlap_count = 0
    for elem in set3:
        count1 = seq1.count(elem)
        count2 = seq2.count(elem)
        overlap_count += min(count1, count2)

    return overlap_count


def is_overlap(subseq, seq):
    # 判断两个集合是否相等
    if set(subseq) == set(seq):
        return True


def match_reward(true_seq, generated_seq):
    """
    计算生成序列与真实数据中1~n长度的症状序列的多种排列方式的重合次数，返回总奖励值
    """
    # 初始化奖励率
    rate = 0.5
    # 初始化长度为n的奖励列表
    reward_list = [0] * len(generated_seq)
    # if len(reward_list)>0:
    #     reward_list[0] = 1
    flag = 1
    # 滑动窗口遍历生成序列
    for wd in range(1, len(generated_seq) + 1):
        if generated_seq[wd - 1] == 0 or len(true_seq) < wd:
            break
        if flag != wd:
            break
        for i in range(len(generated_seq) - wd + 1):
            if generated_seq[i] == 0 or len(true_seq) < i:
                break
            gen_w = generated_seq[i:i + wd]
            # 遍历症状序列的所有子序列，计算与生成序列的重合次数，并更新奖励列表
            for i in range(len(true_seq) - wd + 1):
                subseq = true_seq[i:i + wd]
                overlap_count = count_permutations_overlap(subseq, gen_w)
                if overlap_count >= wd:
                    reward_list[wd - 1] = 1
                    flag += 1
                    break
            # for subseq in itertools.permutations(true_seq, wd):
            #     overlap_count = count_permutations_overlap(subseq, gen_w)
            #     if overlap_count >= wd:
            #         reward_list[wd-1] = 1
            #         flag+=1
            #         break
            # true_w = true_seq[i:i + wd]
            # true_w = set(true_w)
            # gen_w = set(gen_w)
            # # 判断两个set交集的个数
            # if len(true_w.intersection(gen_w))>=wd:
            #     reward_list[wd-1] = 1
            #     # wd_fill = True
            #     flag+=1
            #     break
            # if true_w == gen_w:
            #     reward_list[wd-1] = 1
            #     break
            # for subseq in itertools.permutations(true_seq, wd):
            #
            #     # overlap_count = count_permutations_overlap(subseq, seq)
            #     if is_overlap(subseq, gen_w):
            #         reward_list[wd-1] = 1
            #         wd_fill = True
            #         break

    # for j in range(len(generated_seq)):
    #     seq = generated_seq[:j + 1]

    # 计算总奖励值
    for i in range(len(reward_list)):
        reward_list[i] *= rate
        rate += 0.1

    return reward_list


class RewardDistributor:

    def __init__(self, sv: SymptomVocab, dv: DiseaseVocab, dscom, sscom=None, add_SS=False, is_config=False):

        self.sv = sv
        self.dv = dv

        # 症状恢复奖励
        self.pos_priori_reward = 1.0
        self.neg_priori_reward = -1.0
        # self.hit_reward = {1: 10.0, 2: 5.0, 3: 5.0}
        self.hit_reward = {1: 10.0, 2: 8.0, 3: 5.0}
        self.decline_rate = 0.1
        self.repeat_reward = 0.0
        self.end_reward = 0.0
        # self.missed_reward = -0.2
        self.missed_reward = -1.0
        self.dscom = dscom
        self.sscom = sscom
        self.add_SS = add_SS
        self.is_config = is_config

    '''
    症状与疾病的共现矩阵、症状与症状的共现矩阵同时作为先验奖励时，需要注意两者同时作用后的累加效果
    首先症状-症状共现之间的关系更为复杂，并且症状-症状共现矩阵的稀疏性更高，
    疾病-症状共现矩阵的存在，是为了让智能体不生成无关的症状；症状-症状共现矩阵的存在，是为了让智能体生成相关的症状。在pretrain阶段，自回归的训练方式，
    使得症状的推荐趋于某些固定的序列，因此要通过症状-症状共现矩阵来引导智能体生成相关的症状，而不是某些固定的症状序列。
    仍需注意的是：某个疾病往往对应于一个症状群，当某个症状不与当前疾病共现但与疾病中对应的某些疾病有较高的共现率时，给予正向的reward易将其引导向另一症状群。因此在症状与疾病共现过的基础上，
    再==有区分性==的去对生成的症状进行奖励，即症状-症状共现矩阵的作用。
    '''

    def compute_sr_priori_reward(self, action, dis_id, imp_sx, exp_sx, eps_1=0.0,
                                 eps_2=0.0):  # 先验奖励，push智能体不生成无关的症状（由语料库中的疾病-症状共现矩阵决定）
        reward = []
        # imp_ids = imp_id[1:, :]
        imp_ids = imp_sx
        exp_ids = exp_sx
        if len(action) > 1:
            last_action = action[1]
        else:
            last_action = 1
        if self.is_config:
            for index, act in enumerate(action):
                ds_rate = self.dscom[dis_id, act]
                if ds_rate > eps_1:
                    reward.append(ds_rate * 10)
                    ss_rate = self.sscom[last_action, act]
                    if ss_rate > eps_2:
                        reward[index] += ss_rate * 10
                    last_action = act
                    # reward.append(self.pos_priori_reward)
                    # for sx_id in exp_ids + imp_ids:
                    #     ss_rate = self.sscom[act, sx_id]
                    #     if ss_rate > eps_2:
                    #         reward[index] += ss_rate
                else:
                    reward.append(self.neg_priori_reward)


        else:
            for act in action:
                ds_rate = self.dscom[dis_id, act]
                if ds_rate > eps_1:
                    reward.append(self.pos_priori_reward)
                else:
                    reward.append(self.neg_priori_reward)

        return reward

    def compute_sr_ground_reward(self, action, imp_sx, imp_attr, num_hit):
        # 真实奖励，push智能体生成ground truth中的症状
        # 1. 如果智能体生成了隐形症状中包含的症状，给予正向奖励
        # 2. 如果智能体生成了结束症状，给予奖励等于当前已经命中的症状数减去隐形症状中包含的症状数（该参数设置与智能体的询问轮数由直接关系）
        # 3. 如果智能体生成了隐形症状中不包含的症状，给予负向奖励
        # 4. 如果智能体生成了已询问过的症状，给予负向奖励
        reward = []
        history_acts, num_repeat = set(), 0
        for i, act in enumerate(action):
            if act in history_acts:
                num_repeat += 1
                reward.append(self.repeat_reward)
            else:
                history_acts.add(act)
                if act == self.sv.end_idx:
                    reward.append(num_hit - len(imp_sx) + self.end_reward)
                else:
                    if act in imp_sx:
                        idx = imp_sx.index(act)
                        attr = imp_attr[idx]
                        reward.append(self.hit_reward[attr] - i * self.decline_rate)
                        # reward.append(self.hit_reward[attr])
                    else:
                        reward.append(self.missed_reward)
        return reward, num_repeat

    '''
    pretrain过程中学习到的症状序列关系中,潜在包含了医生的经验，并且这些相关性序列的关系是相互的,即不存在先后性(当然有更为复杂的,有先后影响的情况).
    因此在进行finetune时，
    可以将生成的症状序列与真实数据中1~n长度的症状序列的多种排列方式进行匹配(对于不同长度的症状序列子集,成功一次即给予该长度正向奖励.
    若症状长度=5,则将计算从1~5长度的该序列子集与生成序列的重复(交集不为空),最终计算结果为长度为n的list[1,0,1,1,1],该list映射到reward时,应该*rate,
    该rate从0.5递增).
    '''

    @staticmethod
    def compute_sr_global_reward(action, imp_sx, num_hit, eps=1e-3):
        # 全局奖励，push智能体生成与真实情况下，顺序尽可能类似的序列
        # 1.非序列相关奖励（杰卡德距离）
        set(action).intersection()
        distance = (num_hit + eps) / (len(set(action).union(set(imp_sx))) + eps)
        # 2.序列相关奖励（BLEU，其中denoise action是action和隐形症状序列中的公共子序列）
        denoise_action = [act for act in action if act in imp_sx]

        if is_match:
            distance = [distance if act in imp_sx else 0 for act in action]
            mr = match_reward(imp_sx, action)
            mr = [max(mr) if act in imp_sx else 0 for act in action]

        else:
            bleu = sentence_bleu([imp_sx], denoise_action,
                                 smoothing_function=SmoothingFunction().method1)  # bleu计算1-4元组的匹配度，之后根据权重计算最终的bleu值
            # 这些奖励仅分配到命中的那些症状（）
            distance = [distance if act in imp_sx else 0 for act in action]
            bleu = [bleu if act in imp_sx else 0 for act in action]
            return distance, bleu, denoise_action
        return denoise_action, mr, distance

    # 计算症状恢复奖励（symptom recovery）
    def compute_sr_reward(self, actions, np_batch, epoch, sir: SIRecorder):
        batch_size, seq_len = actions.shape
        # 将 actions 转化为 numpy array
        actions = to_numpy_(actions)
        # 初始化奖励，询问轮数，症状命中数（yes/no/not sure）
        rewards, num_turns, num_hits, num_pos_hits, num_neg_hits, num_ns_hits = [], [], [], [], [], []
        # 初始化重复次数，去噪的动作序列
        num_repeats, denoise_actions, distances, bleus = [], [], [], []
        # 计算每一个序列的回报
        for idx in range(batch_size):
            # break
            # 得到隐形症状序列的yes/no/not sure子序列
            imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx = self.truncate_imp_sx(idx, np_batch)
            # 得到显性
            exp_sx, exp_attr, exp_pos_sx, exp_neg_sx, exp_ns_sx = self.truncate_exp_sx(idx, np_batch)
            # 得到动作序列（通过结束症状id进行截断）
            action = self.truncate_action(idx, actions)
            # 计算询问轮数，症状命中数等
            num_turns.append(len(action)/action_nums)
            num_hit = len(set(action).intersection(set(imp_sx)))
            num_hits.append(num_hit)
            num_pos_hits.append(len(set(action).intersection(set(imp_pos_sx))))
            num_neg_hits.append(len(set(action).intersection(set(imp_neg_sx))))
            num_ns_hits.append(len(set(action).intersection(set(imp_ns_sx))))
            # 计算先验奖励
            priori_reward = self.compute_sr_priori_reward(action, dis_id=np_batch['labels'][idx], imp_sx=imp_sx,
                                                          exp_sx=exp_sx)
            # 计算真实奖励
            ground_reward, num_repeat = self.compute_sr_ground_reward(action, imp_sx, imp_attr, num_hit)
            num_repeats.append(num_repeat)
            # 计算全局奖励
            if is_match:
                denoise_action, mr, distance = self.compute_sr_global_reward(action, imp_sx, num_hit)
                reward = [pr + gr + mr + dr for pr, gr, mr, dr in zip(priori_reward, ground_reward, mr, distance)]
                distances.append(0)
                bleus.append(0)
            else:
                distance, bleu, denoise_action = self.compute_sr_global_reward(action, imp_sx, num_hit)
                distances.append(0 if len(distance) == 0 else max(distance))
                bleus.append(0 if len(bleu) == 0 else max(bleu))
                reward = [pr + gr + dr + br for pr, gr, dr, br in zip(priori_reward, ground_reward, distance, bleu)]
            denoise_actions.append(denoise_action)
            # 计算最终奖励（每一步的奖励）

            reward += [0] * (seq_len - len(action))
            rewards.append(reward)
        sir.update(rewards, num_turns, num_hits, num_pos_hits, num_neg_hits,
                   num_ns_hits, num_repeats, distances, bleus, epoch=epoch)
        return rewards

    def truncate_action(self, idx: int, actions: list) -> list:
        action = actions[idx].tolist()
        return action[: action.index(self.sv.end_idx)] if self.sv.end_idx in action else action

    def truncate_imp_sx(self, idx: int, np_batch: dict):
        imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx = [], [], [], [], []
        for sx_id, attr_id in zip(np_batch['imp_sx_ids'][1:, idx], np_batch['imp_attr_ids'][1:, idx]):
            if sx_id == self.sv.end_idx:
                break
            else:
                imp_sx.append(sx_id)
                imp_attr.append(attr_id)
                if attr_id == self.sv.SX_ATTR_POS_IDX:
                    imp_pos_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NEG_IDX:
                    imp_neg_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NS_IDX:
                    imp_ns_sx.append(sx_id)
        return imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx

    # 截取有效的explicit symptoms
    def truncate_exp_sx(self, idx: int, np_batch: dict):
        exp_sx, exp_attr, exp_pos_sx, exp_neg_sx, exp_ns_sx = [], [], [], [], []
        for sx_id, attr_id in zip(np_batch['exp_sx_ids'][:, idx], np_batch['exp_attr_ids'][:, idx]):
            if sx_id == self.sv.SX_ATTR_PAD_IDX:
                break
            else:
                exp_sx.append(sx_id)
                exp_attr.append(attr_id)
                if attr_id == self.sv.SX_ATTR_POS_IDX:
                    exp_pos_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NEG_IDX:
                    exp_neg_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NS_IDX:
                    exp_ns_sx.append(sx_id)
        return exp_sx, exp_attr, exp_pos_sx, exp_neg_sx, exp_ns_sx


def make_features_neural(sx_ids, attr_ids, labels, sv: SymptomVocab):
    from conf import suffix
    feats = []
    for sx_id, attr_id in zip(sx_ids, attr_ids):
        feature = []
        sample = sv.decoder(sx_id, attr_id)
        for sx, attr in sample.items():
            feature.append(sx + suffix.get(attr))
        feats.append(' '.join(feature))
    return feats, labels


def extract_features(sx_ids, attr_ids, sv: SymptomVocab):
    sx_feats, attr_feats = [], []
    for idx in range(len(sx_ids)):
        sx_feat, attr_feat = [sv.start_idx], [sv.SX_ATTR_PAD_IDX]
        for sx_id, attr_id in zip(sx_ids[idx], attr_ids[idx]):
            if attr_id not in [sv.SX_ATTR_PAD_IDX, sv.SX_ATTR_NM_IDX]:  # 分隔explicit与implicit的start对应的attr为0因此被去除
                # 去除无效的症状和属性pairs
                sx_feat.append(sx_id)
                attr_feat.append(attr_id)
        sx_feats.append(to_tensor_(sx_feat))
        attr_feats.append(to_tensor_(attr_feat))
    return sx_feats, attr_feats


def make_features_xfmr(sv: SymptomVocab, batch, si_sx_ids=None, si_attr_ids=None, merge_act: bool = False,
                       merge_si: bool = False):
    # convert to numpy
    assert merge_act or merge_si
    sx_feats, attr_feats = [], []
    if merge_act:
        act_sx_ids = torch.cat([batch['exp_sx_ids'], batch['imp_sx_ids']]).permute([1, 0])
        act_attr_ids = torch.cat([batch['exp_attr_ids'], batch['imp_attr_ids']]).permute([1, 0])
        act_sx_ids, act_attr_ids = to_numpy_vla(act_sx_ids, act_attr_ids)
        act_sx_feats, act_attr_feats = extract_features(act_sx_ids, act_attr_ids, sv)
        sx_feats += act_sx_feats
        attr_feats += act_attr_feats
    if merge_si:
        si_sx_ids, si_attr_ids = to_numpy_vla(si_sx_ids, si_attr_ids)
        si_sx_feats, si_attr_feats = extract_features(si_sx_ids, si_attr_ids, sv)
        sx_feats += si_sx_feats
        attr_feats += si_attr_feats
    sx_feats = pad_sequence(sx_feats, padding_value=sv.pad_idx).long()
    attr_feats = pad_sequence(attr_feats, padding_value=sv.SX_ATTR_PAD_IDX).long()
    return sx_feats, attr_feats
