import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=10,n_output=1)

#torch.optim是优化器，里面要net的参数net.parameters和学习率lr
optimizer = torch.optim.SGD(net.parameters(),lr = 0.2)

#Loss
loss_func = nn.MSELoss()

#plt interactive mode on/off
plt.ion()

#torch.unsqueeze(数据, dim=要加维度的位置)
x = torch.linspace(-1,1,100)
x = torch.unsqueeze(x,dim=1)
#torch.rand(shape)生成形状的随机张量
ran = 0.2*torch.rand(x.size())
y = x.pow(2) + ran


for i in range(10000):
    #输入随机值
    prediction = net(x)
    #损失函数
    loss = loss_func(prediction,y)
    #必须清理梯度
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #更新权重矩阵
    optimizer.step()

    if i % 5 == 0:
        #清理屏幕
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),c='r',lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()

