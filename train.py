import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Speed_Prediction, Speed_Prediction_with_mass
from csv_dataloader import load_csv_to_dataloader
#length=
#ms
#g
# 保存检查点
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename, norm_params=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'norm_params': norm_params
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")
    
def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from {filename}, resuming at epoch {epoch}")
    return model, optimizer, epoch, train_loss, val_loss

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path, norm_params, resume=False):
    model = model.to(device)
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 恢复训练（如果需要）
    if resume and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path, device)
        model = model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存检查点（每 10 epoch 或最佳模型）
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, 
                           f"ckpt/checkpoint_epoch_{epoch+1}.pth", norm_params)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, 
                           "ckpt/best_model.pth", norm_params)
            
# 主函数
def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="Train Speed Prediction Model")
    parser.add_argument('--csv_file', type=str, default='data/all_data.csv', 
                        help='Path to CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=800, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--val_split', type=float, default=0.2, help='the ratio of validation set')
    parser.add_argument('--checkpoint_path', type=str, default='ckpt/best_model.pth', 
                        help='Path to checkpoint')
    parser.add_argument('--length', type=float, default=0.991, help='Length of the object')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_loader, val_loader, norm_params = load_csv_to_dataloader(args.csv_file, args.length, args.batch_size, val_split = args.val_split)
    
    # 初始化模型、损失函数和优化器
    model = Speed_Prediction(input_orders=4)  # 或者 Speed_Prediction_with_mass(input_orders=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练
    train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                args.num_epochs, args.checkpoint_path, norm_params, args.resume)

if __name__ == "__main__":
    main()