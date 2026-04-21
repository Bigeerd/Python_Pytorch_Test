import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.out = nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x
net = Net(n_feature=2,n_hidden=10,n_output=2)

n_data = torch.ones(100,2)#生成全1形状的张量
x0 = torch.normal(n_data*2,1)#输入张量按照标准差高斯分布随机以下
y0 = torch.zeros(100)#生成全0矩阵
x1 = torch.normal(n_data*-2,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)#dim=0竖着拼，dim=1横着拼
y = torch.cat((y0,y1),0).type(torch.LongTensor)
print(x)
print(y)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = nn.CrossEntropyLoss()

plt.ion()

for t in range(100):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')


        plt.pause(0.1)


plt.ioff()
plt.show()