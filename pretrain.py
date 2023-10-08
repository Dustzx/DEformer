from torch.utils.data import DataLoader

from utils import load_data

from layers import Agent
from utils import make_dirs
# from sklearn.metrics import accuracy_score

from data_utils import *
from conf import *
####################
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
import functools

# 创建记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器
log_file = f"log/pre_train/{train_dataset}/log_{datetime.now().strftime('%Y_%m_%d_%H')}.txt"
# 创建目录
make_dirs(log_file)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 创建 TensorBoardX 摘要写入器
tb_writer = SummaryWriter(log_dir="logs\\pre_train\\{}".format(train_dataset))
####################
train_samples, test_samples = load_data(train_path), load_data(test_path)
train_size, test_size = len(train_samples), len(test_samples)

sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)

# pretrain中增加两个共现矩阵在DataLoader的collate_fn中发挥作用
# compute disease-symptom co-occurrence matrix
# global dscom
dscom = compute_dscom(train_samples, sv, dv, smooth=True)
# compute symptom-symptom co-occurrence matrix
# global sscom
# global median_list
sscom, median_list, avg_list = compute_sscom(train_samples, sv, normalize=False)
# 将median_list中nan值替换为无穷大
for i in range(len(median_list)):
    if np.isnan(median_list[i]):
        median_list[i] = float('inf')
    if np.isnan(avg_list[i]):
        avg_list[i] = float('inf')
flag = f"===========================is_position:{dec_add_pos},is_shuffle:{is_shuffle},full_shuffle:{full_shuffle}==========================="
logger.info(flag)
# global avg_ss
num_sxs, num_dis = sv.num_sxs, dv.num_dis
# avg_ss = [sum/num_sxs for sum in row_sums]
collate_fn = functools.partial(lm_collater_shuffle, sscom=sscom, median_list=median_list, avg_list=avg_list)

train_ds = SymptomDataset(train_samples, sv, dv, keep_unk=False, add_tgt_start=True, add_tgt_end=True)
train_ds_loader = DataLoader(train_ds, batch_size=train_bsz, num_workers=num_workers, shuffle=True,
                             collate_fn=collate_fn)

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=test_bsz, num_workers=num_workers, shuffle=False,
                            collate_fn=lm_collater)

model = Agent(num_sxs, num_dis).to(device)

si_criterion = torch.nn.CrossEntropyLoss(ignore_index=sv.pad_idx).to(device)
dc_criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=pt_learning_rate, weight_decay=pt_weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
make_dirs([best_pt_path, last_pt_path])

best_acc = 0
print('pre-training...')


def pretrain(num_repeats=1):
    for num in range(num_repeats):
        best_acc = 0
        last_pt_path = 'saved/{}/last_pt_model_{}.pt'.format(train_dataset, num)
        for epoch in range(pt_train_epochs):
            # break
            train_loss, train_si_loss, train_dc_loss = [], [], []
            train_num_hits, test_num_hits = 0, 0
            model.train()
            for batch in train_ds_loader:
                # break
                sx_ids, attr_ids, labels = batch['sx_ids'], batch['attr_ids'], batch['labels']
                seq_len, bsz = sx_ids.shape
                shift_sx_ids = torch.cat([sx_ids[1:], torch.zeros((1, bsz), dtype=torch.long, device=device)], dim=0)
                # symptom inquiry
                mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
                si_outputs = model.symptom_decoder.get_features(model.symptom_decoder(sx_ids, attr_ids, mask=mask))
                si_loss = si_criterion(si_outputs.view(-1, num_sxs), shift_sx_ids.view(-1))
                # disease classification
                # pt_warmup_epochs = pt_train_epochs // 2 if is_shuffle else -1
                if epoch > pt_warmup_epochs:
                    si_sx_feats, si_attr_feats = make_features_xfmr(
                        sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
                    dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                    dc_loss = dc_criterion(dc_outputs, batch['labels'])
                    loss = si_loss + dc_loss
                    # record
                    train_loss.append(loss.item())
                    train_si_loss.append(si_loss.item())
                    train_dc_loss.append(dc_loss.item())
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    train_num_hits += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()


                else:
                    loss = si_loss
                    # record
                    train_loss.append(loss.item())
                    train_si_loss.append(si_loss.item())
                    train_dc_loss.append(0)
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
            train_acc = train_num_hits / train_size
            #########
            logger.info(f"Epoch [{epoch + 1}/{pt_train_epochs}] - Loss: {loss:.4f}")
            tb_writer.add_scalar("train_loss", loss, epoch + 1)
            ########
            model.eval()
            for batch in test_ds_loader:
                sx_ids, attr_ids, labels = batch['sx_ids'], batch['attr_ids'], batch['labels']
                si_sx_feats, si_attr_feats = make_features_xfmr(
                    sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
                dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                test_num_hits += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
            test_acc = test_num_hits / test_size
            if test_acc > best_acc:
                best_acc = test_acc
                model.save(last_pt_path)
            # print('epoch: {}, train total/si/dc loss: {}/{}/{}, train/test/best acc: {}/{}/{}'.format(
            #     epoch + 1, np.round(np.mean(train_loss), digits), np.round(np.mean(train_si_loss), digits),
            #     np.round(np.mean(train_dc_loss), digits), round(train_acc, digits),
            #     round(test_acc, digits), round(best_acc, digits)))

            logger.info('epoch: {}, train total/si/dc loss: {}/{}/{}, train/test/best acc: {}/{}/{}\n'.format(
                epoch + 1, np.round(np.mean(train_loss), digits), np.round(np.mean(train_si_loss), digits),
                np.round(np.mean(train_dc_loss), digits), round(train_acc, digits),
                round(test_acc, digits), round(best_acc, digits)))
        logger.info(f'====================pt_warmup_epochs{pt_warmup_epochs}======================\n')
        # model.save(last_pt_path)


pretrain(num_repeats=1)
################################################################
logger.removeHandler(file_handler)
logger.removeHandler(console_handler)
file_handler.close()
tb_writer.close()
