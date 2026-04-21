import torch
import matplotlib.pyplot as plt

#得到start到end中间划分为step步的张量
def linespace(start,end,step):
    return torch.linspace(start,end,step)
#在dim位置变化tensor
def squeeze(tensor,dim):
    return torch.unsqueeze(tensor,dim)
#生成shape形状的tensor，里面每一个量都是随机的
def random_tensor(shape):
    return torch.rand(shape)
#把tensor里的每一个值平方
def pow_tensor(tensor):
    return tensor.pow(2)
#生成一个编号为number，大小为x_size*y_size的图
def generate_figure(number,x_size,y_size):
    plt.figure(number,figsize=(x_size,y_size))
#画出一个x轴为x_axis，y轴为y_axis的散点图
def draw_scatter(x_axis,y_axis):
    plt.scatter(x_axis,y_axis)
#画出一个x轴为x_axis，y轴为y_axis的折线图
def draw_plot(x_axis,y_axis):
    plt.plot(x_axis,y_axis)
#生成一个input,mid,output的linear+relu+linear的net
def sequential(input,mid,output):
    return torch.nn.Sequential(
        torch.nn.Linear(input,mid),
        torch.nn.ReLU(),
        torch.nn.Linear(mid,output)
    )

#data
x = torch.unsqueeze(torch.linspace(-1,1,100),1)
y = x.pow(2) + torch.rand(x.size())*0.2

#保存参数
def save_net():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1,figsize=(5,5))
    plt.subplot(131)
    plt.title("Net1")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy())

    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),"net_params.pkl")

#
def restore_net():
    net2 = torch.load("net.pkl",weights_only=False)
    prediction = net2(x)

    plt.subplot(132)
    plt.title("Net2")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy())

def restore_net_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    net3.load_state_dict(torch.load("net_params.pkl"))

    prediction = net3(x)

    plt.subplot(133)
    plt.title("Net3")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy())


save_net()
restore_net()
restore_net_params()
plt.show()