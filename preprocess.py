import torch

def preprocess_for_prediction(v0, normalization_params):
    """
    对预测输入进行与训练时相同的预处理
    
    参数:
        v0: 初始速度(浮点数)
        normalization_params: 标准化参数字典
    
    返回:
        处理后的特征张量，可直接输入模型
    """
    # 生成高阶特征
    order = 4
    features = torch.tensor([v0**i for i in range(1, order + 1)], dtype=torch.float32).reshape(1, -1)
    
    # 应用相同的标准化
    mean = normalization_params['mean']
    std = normalization_params['std']
    normalized_features = (features - mean) / std
    
    return normalized_features