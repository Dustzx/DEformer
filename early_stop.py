from torch.utils.data import DataLoader
from utils import load_data
from layers import Agent
from tqdm import tqdm
import numpy as np

from data_utils import *
from conf import *

# load dataset
train_samples, test_samples = load_data(train_path), load_data(test_path)
test_size = len(test_samples)

# construct symptom & disease vocabulary
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=pg_collater)

# compute disease-symptom co-occurrence matrix
dscom = compute_dscom(train_samples, sv, dv)

# global median_list
sscom, mid_list, avg_list = compute_sscom(train_samples, sv, normalize=False)
# 将median_list中nan值替换为无穷大
for i in range(len(mid_list)):
    if np.isnan(mid_list[i]):
        mid_list[i] = float('inf')
    if np.isnan(avg_list[i]):
        avg_list[i] = float('inf')
# init reward distributor
rd = RewardDistributor(sv, dv, dscom)

# init patient simulator
ps = PatientSimulator(sv)

# init agent
model = Agent(num_sxs, num_dis).to(device)

max_turn = num_turns
# dense_one
model_name = 'metric_model_{}.pt'.format('40_1')
# model_name = 'acc_model_{}.pt'.format('27_4')
metric_model_path = 'saved/{}/dense_all/{}'.format(train_dataset, model_name)
# metric_model_path = 'saved/all/best_pt_model.pt'
model.load(metric_model_path)
print('load model from {}......'.format(metric_model_path))
flag = f"===========================pt_model:{pt_path},data_set:{train_dataset},model_name:{model_name},is_position:{dec_add_pos},is_shuffle:{is_shuffle},is_config:{is_config}==========================="

# 当症状的不确定性很大，确定性很小
# 疾病的确定性很大，确定性很大

# eps = [0.5, 0.6, 0.7, 0.8, 0.850,0.860,0.870,0.88,0.89,0.90,0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0]
eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.6, 0.62, 0.65, 0.66, 0.7, 0.8, 0.82, 0.83,
       0.84, 0.850, 0.860, 0.870, 0.88, 0.89,
       0.90, 0.91, 0.915, 0.92,
       0.925, 0.93, 0.935, 0.94, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0]
# eps = [0.85,0.86,0.87,0.875,0.88,0.885,0.89]
# eps = [0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0]
# eps = [0.5, 0.6, 0.65, 0.7, 0.73, 0.74, 0.75, 0.76, 0.8, 0.85, 0.9, 0.95, 1.0]
# recall = [0.5, 0.6, 0.65, 0.7, 0.73, 0.74, 0.75, 0.76, 0.8, 0.85, 0.9, 0.95, 1.0]
recall = [0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.6, 0.62, 0.65, 0.66, 0.7, 0.8, 0.82,
          0.83, 0.84, 0.850, 0.860, 0.870, 0.88, 0.89,
          0.90, 0.91,
          0.915, 0.92,
          0.925, 0.93, 0.935, 0.94, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0]
miss_num = [12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 3, 3, 2, 2, 5, 5, 5, 5]

# imp_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# e = 0
# imp_num = [11,14,17]
turns = []
accs = []
rc = 0
with open('early_log.txt', 'a') as f:
    f.write(str(flag) + '\n')
idx = 0


def early_s(e, rc, r, miss_num=6, si_num=12):
    test_num_hits = 0
    actual_turns = []
    max_probs = []
    exp_end_nums = 0
    imp_rcs = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_ds_loader):
            # if rand_num:
            #     random_num = random.randint(3, 12)
            # else:
            #     random_num = 10
            # print(random_num)
            actual_turn, is_success, max_prob, exp_end, imp_recall = model.execute(batch, ps, sv, max_turn, e,
                                                                                   rc, r, miss_num, si_num)
            test_num_hits += is_success
            actual_turns.append(actual_turn)
            max_probs.append(np.round(max_prob, digits))
            exp_end_nums += exp_end
            imp_rcs += imp_recall
    test_acc = test_num_hits / test_size
    avg_turn = np.mean(actual_turns)
    turns.append(avg_turn)
    accs.append(test_acc)
    if imp_rcs != 0:
        imp_recalls = imp_rcs / (test_size - exp_end_nums)
        imp_rc_real = imp_rcs / test_size
    else:
        imp_recalls = 0
        imp_rc_real = 0

    result = 'eps: {}, avg turn: {}, acc: {}, recall: {},recall_r:{},exp_end_nums:{}, recall_threshold:{}.'.format(
        e, np.round(np.mean(actual_turns), digits), np.round(test_acc, digits), np.round(imp_recalls, digits),
        np.round(imp_rc_real, digits),
        exp_end_nums, rc)
    # 将result保存到log.txt中
    with open('early_log.txt', 'a') as f:
        f.write(str(result) + '\n')
    print(result)
    return test_acc, imp_rc_real


# # 随机值测试模拟的recall
#
# for e, rc in tqdm(zip(eps, recall)):
#
#     early_s(e=e, rc=rc, r=2,rand_num=True)
# 固定值测试模拟的recall
# for i in range(15, 10,-1):
# for i in range(10, 5, -2):
# for e, rc in tqdm(zip(eps, recall)):
#     early_s(e=0, rc=rc, r=2, si_num=15, miss_num=6)

# for e, rc in tqdm(zip(eps, recall)):
#     early_s(e=e, rc=rc, r=2)
# 对于使用miss_num、si_num的转换准则，需要单独测试
def si_test():
    # 基于rc+si_num+miss_num
    for e, rc in tqdm(zip(eps, recall)):
        early_s(e=0, rc=rc, r=2, si_num=15, miss_num=10)
    # 基于e+miss_num
    for e in tqdm(eps):
        early_s(e=e, rc=0, r=1, miss_num=8)
def ruler_run():
    # r1、r2、r3、r4
    for r in range(4):
        repeat = 0
        last_acc = 0
        last_rc = 0
        with open('early_log.txt', 'a') as f:
            f.write(f'----------------------{r + 1}-------------------------' + '\n')
        if r == 0:
            # r1 完全基于症状回归率的转换准则
            for rc in tqdm(recall):
                acc, rc = early_s(e=0, rc=rc, r=r)
                if acc == last_acc and rc == last_rc:
                    repeat+=1
                last_acc = acc
                last_rc = rc
                if repeat>=4:
                    break
        if r == 1:
            # r2 基于模拟症状回归率的转换准则
            for rc in tqdm(recall):
                acc, rc = early_s(e=0, rc=rc, r=r, miss_num=6, si_num=18)  # dxybest 6/15
                if acc == last_acc and rc == last_rc:
                    repeat+=1
                last_acc = acc
                last_rc = rc
                if repeat>=4:
                    break
        elif r == 2:
            # r3 改进的DxFormer中的转换准则
            for e in tqdm(eps):
                early_s(e=e, rc=0, r=r, miss_num=8)  # dxybest 6
        elif r == 3:
            # r4 DxFormer中的转换准则
            for e in tqdm(eps):
                early_s(e=e, rc=0, r=r)

# si_test()
ruler_run()
# miss_num、si_num
# dxy 6/15  mz4     mz10
# for e in tqdm(eps + recall):
#
#     if idx > 13:
#         rc = e
#         e = 0
# for recall_threshold in recall:

# eps: 1.0, avg turn: 10.0, acc: 0.7536.
# eps: 0.99, avg turn: 8.0696, acc: 0.742.
# eps: 1.0, avg turn: 20.0, acc: 0.6942.
# eps: 0.99, avg turn: 14.966, acc: 0.6869.
# eps: 0.95, avg turn: 10.3131, acc: 0.6408.

# r2 基于模拟症状回归率的转换准则
# 将eps和recall组合，同时传入
# for rc in tqdm(recall):
#     acc, rc = early_s(e=0, rc=rc, r=1, miss_num=6, si_num=12)  # dxybest 6/15
#     if acc == last_acc and rc == last_rc:
#         break
#     last_acc = acc
#     last_rc = rc
# dec/enc eps: 1.0/0.0, avg turn: 20.0, acc: 0.6796, avg dec/enc entropy: 0.7219/0.173
# dec/enc eps: 1.0/0.005, avg turn: 16.6748, acc: 0.6772, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.01, avg turn: 14.1092, acc: 0.6699, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.02, avg turn: 10.6942, acc: 0.6456, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.03, avg turn: 8.0825, acc: 0.6165, avg dec/enc entropy: 0.6294/0.173
# dec/enc eps: 1.0/0.04, avg turn: 6.9442, acc: 0.6044, avg dec/enc entropy: 0.6294/0.173
# dec/enc eps: 1.0/0.05, avg turn: 5.8374, acc: 0.585, avg dec/enc entropy: 0.6294/0.1667
# dec/enc eps: 1.0/0.08, avg turn: 2.7718, acc: 0.5534, avg dec/enc entropy: 0.5923/0.1667
# dec/enc eps: 1.0/0.1, avg turn: 1.4417, acc: 0.5291, avg dec/enc entropy: 0.5923/0.1667


# entropys = []
# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_ds_loader):
#         entropys.append(model.execute(batch, ps, sv, max_turn))
#
# ess = []
# for i in range(max_turn):
#     es = sum([entropy[i] for entropy in entropys])
#     ess.append(es)
