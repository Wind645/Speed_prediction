import torch
import torch.nn as nn
import torch.nn.functional as F

class Speed_Prediction(nn.Module):
    '''
    这个模型是个速度预测模型，是个简单的小网络，它用于接收物体当前的状态（速度）
    信息，预测物体在一段距离之后（发生碰撞时）的速度。
    
    这里利用了神经网络的天然的非线性本质，尝试去预测物体移动过程中受到的复杂的空气
    阻力等外部影响因素，然后给出物体一定一段距离之后的真实速度。
    
    期望输入：
    - 物体当前的速度
    - 物体当前的速度的二次方
    - 物体当前的速度的三次方
    - 物体当前的速度的四次方
    (可以继续增加高次方的速度值)
    
    输出：
    - 物体在一段距离之后的速度（这个距离是固定的）
    '''
    def __init__(self, input_orders=4):
        
        super(Speed_Prediction, self).__init__()
        self.fc1 = nn.Linear(input_orders, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        '''
        前向传播函数
        x的size为（B, 4），B为batch size，4即存储了v的一次、二次、三次、四次方值
        
        一个潜在的问题在于这里有点违反iid假设，不过忽视这个假设使用神经网络的操作多了去了……
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class Speed_Prediction_with_mass(nn.Module):
    '''
    这个模型是个速度预测模型，是个简单的小网络，它用于接收物体当前的状态（速度，包括质量）
    信息，预测物体在一段距离之后（发生碰撞时）的速度。
    
    这里利用了神经网络的天然的非线性本质，尝试去预测物体移动过程中受到的复杂的空气
    阻力等外部影响因素，然后给出物体一定一段距离之后的真实速度。
    
    期望输入：
    - 物体当前的速度
    - 物体当前的速度的二次方
    - 物体当前的速度的三次方
    - 物体当前的速度的四次方
    (可以继续增加高次方的速度值)
    
    输出：
    - 物体在一段距离之后的速度（这个距离是固定的）
    '''
    def __init__(self, input_orders=4):
        
        super(Speed_Prediction, self).__init__()
        self.fc1 = nn.Linear(input_orders + 1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        '''
        前向传播函数
        x的size为（B, 4），B为batch size，4即存储了物体配重质量，v的一次、二次、三次、四次方值
        
        一个潜在的问题在于这里有点违反iid假设，不过忽视这个假设使用神经网络的操作多了去了……
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
        
        