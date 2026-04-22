import torch,numpy as np,matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = 32,
            num_layers = 1,
            batch_first = True,
        )

        self.out = torch.nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state = self.rnn(x,h_state)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state

rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(100):
    start = step*np.pi
    end = (step+1)*np.pi

    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(np.array(x_np).reshape(1,10,1))
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()