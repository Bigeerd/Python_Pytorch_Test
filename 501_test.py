import torch
import numpy as np,matplotlib.pyplot as plt

INPUT_SIZE = 1
LR = 0.02

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(32,1)
    def forward(self,x,h_state):
        r_out,h_state = self.rnn(x,h_state)

        outs = []
        for time_step in range(x.size(1)):
            outs.append(self.linear(r_out[:,time_step,:]))
        return torch.stack(outs, dim=1),h_state
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

h_state = None
plt.figure(1, figsize=(12, 5))
plt.ion()
step = 0

for i in range(1000):
    dynamic_steps = np.random.randint(1, 4)

    start, end = step * np.pi, (step + dynamic_steps) * np.pi

    step += 1

    steps = np.linspace(start, end, 10*dynamic_steps, dtype=np.float32)

    print(len(steps))

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
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