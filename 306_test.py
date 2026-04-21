import torch
import matplotlib.pyplot as plt

EPOCH = 12
BATCH_SIZE = 32
LR = 0.01

x = torch.unsqueeze(torch.linspace(-1,1,1000),1)
y = x.pow(2) + torch.normal(torch.zeros(x.size()))*0.1



torch_dataset = torch.utils.data.TensorDataset(
    x,
    y,
)
loader = torch.utils.data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.f1 = torch.nn.Linear(1,20)
        self.f2 = torch.nn.Linear(20,1)

    def forward(self,x):
        x = self.f1(x)
        x = torch.nn.functional.relu(x)
        x = self.f2(x)
        return x

if __name__ == "__main__":
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR,alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR,betas=(0.9,0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[],[],[],[]]

    for epoch in range(EPOCH):
        for step,(b_x,b_y) in enumerate(loader):
            for net,opt,l_his in zip(nets,optimizers,losses_his):
                output = net(b_x)
                loss = loss_func(output,b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()