import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from model import Speed_Prediction
from preprocess import preprocess_for_prediction

ckpt_path = 'ckpt\\best_model.pth'
model = Speed_Prediction()
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
device = torch.device('cpu')
v_list = [0.2 + i * 0.02 for i in range(0, 21)]
v_output_list = []
for v in v_list:
    v_input = preprocess_for_prediction(v, state_dict['norm_params'])
    v_input = v_input.to(device)
    v_output = model(v_input)
    print(f"Predicted speed for v={v}: {v_output.item()}")
    v_output_list.append(v_output.item())

sns.set_theme(style="darkgrid")
sns.lineplot(x=v_list, y=v_output_list, marker='o')
sns.lineplot(x=v_list, y=v_list, color='green', linestyle='--', label='y=x')
sns.scatterplot(x=v_list, y=v_output_list, color='red')

plt.show()
    