import torch
import torchvision
import matplotlib.pyplot as plt
import os

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

if not os.path.exists("./mnist_001/") or not os.listdir("./mnist_001/"):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root="./mnist_001/",
    train=True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

# print(train_data.data.shape)
# print(train_data.targets.shape)
# plt.imshow(train_data.data[0].numpy(),cmap='gray')
# plt.title("%i" % train_data.targets[0])
# plt.show()

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(
    root="./mnist_001/",
    train=False,
    transform = torchvision.transforms.ToTensor(),
)

test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets.numpy()[:2000]

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = torch.nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.view(-1,28,28)
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)




