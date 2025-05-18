import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_csv_to_dataloader(csv_file, batch_size=32, shuffle=True, order=4, val_split=0.2):
    '''
    从csv文件加载数据，生成特征(v, v², v³, v⁴)，返回 DataLoader。
    '''
    df = pd.read_csv(csv_file)
    
    # 检查数据完整性
    if df.isnull().any().any():
        raise ValueError("CSV contains missing values")
    if not {'v0', 'v_final'}.issubset(df.columns):
        raise ValueError("CSV must contain 'v0' and 'v_final' columns")
    
    # 提取初始速度和终止速度
    v0 = df['v0'].values
    v_final = df['v_final'].values
    
    # 生成特征：v, v², v³, v⁴
    features = torch.tensor([v0**i for i in range(1, order + 1)], dtype=torch.float32).T
    # (N, 4)
    
    # 目标：终止速度
    targets = torch.tensor(v_final, dtype=torch.float32).reshape(-1, 1)
    
    # 标准化特征（可选，提高训练稳定性）
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    # 划分训练集和验证集
    train_features, val_features, train_targets, val_targets = train_test_split(
        features,targets, test_size=val_split, random_state=42
    )
    
    # 创建 TensorDataset
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader