from layers import Agent
from utils import load_data
from data_utils import *
from conf import *
train_samples, test_samples = load_data(train_path), load_data(test_path)
train_size, test_size = len(train_samples), len(test_samples)
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis
model = Agent(num_sxs, num_dis).to(device)
pretrained_dict = torch.load(best_pt_path)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print(pretrained_dict.keys())
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)