import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#np.arange(start, stop, step, dtype=None)
#a = np.arange(start=11,stop=1,step=-1,dtype=int)
#print(a)


#np.range(start, stop, step)
#a = []
#for i in range(1,10,3):
#    a.append(i)
#print(a)

#np.reshape(行数, 列数)
#print(np.arange(start=1,stop=10,step=1).reshape(3,3))
#arr = [1,2,3,4,5,6]
#arr = np.array(arr)
#print(arr.reshape(-1,2))

#torch.from_numpy(ndarray)
#所有的np
# arr = (1,2,3,4,5,6)
# arr = np.array(arr)
# print(arr)
# print("\n")
# arr = arr.reshape(-1,2)
# arr1 = torch.from_numpy(arr)
# arr2 = torch.tensor([1,2,3,4,5,6])
# print(arr1)
# print(arr2)


#np.array(object, dtype=None)


#张量.numpy()
# arr2 = torch.tensor([1,2,3,4,5,6])
# print(arr2)
# arr2 = arr2.numpy()
# print(arr2)

#torch.FloatTensor(数据)
# arr1 = torch.FloatTensor([-1, 2, 3])
# print(arr1)
# print(type(arr1))
# arr1 = torch.LongTensor([-1, 2, 3])
# print(arr1)
# print(type(arr1))

#np.abs(x)
# arr = np.array([-1, 2, 3])
# arr1 = np.abs(arr)
# print(arr1)
# print(type(arr1))

# arr = torch.tensor([-1, 2, 3])
# arr1 = torch.abs(arr)
# print(arr1)

#np.mean(x)
# data = [-1,-2,1,2]
# arr = np.array(data)
# print(np.mean(arr))
# arr = torch.Tensor(arr)
# print(torch.mean(arr))

#np.sin(x)
#torch.sin(tensor)
# data = [-1,-2,1,2]
# print(np.sin(data))
# print(torch.sin(torch.tensor(data)))

#
# data = [3,4]
# data = np.array(data)
# print(data.dot(data))
# data = torch.Tensor(data)
# print(data.dot(data))


# print(np.dot(np.array([1,2]),np.array([3,4])))
# print(np.dot(np.array([[1,1],[1,1]]),np.array([[1,1],[1,1]])))
# print(torch.dot(torch.tensor([1,2]),torch.tensor([3,4])))
#
#
# #数组1.dot(数组2)
# print(np.array([3,4]).dot(np.array([1,2])))
# #张量1.dot(张量2)
# print(torch.tensor([3,4]).dot(torch.tensor([1,2])))

# #np.matmul(a, b)
# print(np.matmul([1,1],[[1],[1]]))
#
# #torch.mm(tensor_a, tensor_b)
# a = torch.tensor([[1,1],[1,1]])
# print(torch.mm(a,a))

# tensor = torch.Tensor([[1,2],[3,4]])
# variable = Variable(tensor, requires_grad=True)
# v_out = torch.mean(variable*variable*variable)
# print(f"v_out:{v_out}")
#
# print(v_out.backward())
#
# print(variable)
# #.backward对输入数值的每一个可能贡献的数字求导,.grad查看backward求导之后的结果
# print(variable.grad)
# print(variable.data)
# print(variable.data.numpy())

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

#过relu层
x_relu = torch.relu(x).data.numpy()

x_sigmoid = torch.sigmoid(x).data.numpy()

x_tanh = torch.tanh(x).data.numpy()

#softplus(x) = ln( 1 + eˣ )
x_softplus = F.softplus(x).data.numpy()

x_softmax = torch.softmax(x, dim=0).data.numpy()

#新建画布，编号和大小
plt.figure(1,figsize=(5,5))
#对哪块进行操作。nxn，第几块
plt.subplot(231)
#x轴，y轴，颜色和label
plt.plot(x_np,x_relu,c='red',label='relu')
#控制y轴的显示范围
plt.ylim((-1,5))
#显示label，loc自己选best
plt.legend(loc='best')

plt.subplot(232)
plt.plot(x_np,x_sigmoid,c='blue',label='sigmoid')
plt.legend(loc='best')

plt.subplot(233)
plt.plot(x_np,x_tanh,c='green',label='tanh')
plt.legend(loc='best')

plt.subplot(234)
plt.plot(x_np,x_softplus,c='magenta',label='softplus')
plt.legend(loc='best')

plt.subplot(235)
plt.plot(x_np,x_softmax,c='cyan',label='softmax')
plt.legend(loc='best')

plt.show()
