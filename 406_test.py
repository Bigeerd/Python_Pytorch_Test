import torch
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():
    a = np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    paintings = a * np.power(PAINT_POINTS,2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

G = torch.nn.Sequential(
    torch.nn.Linear(N_IDEAS,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,ART_COMPONENTS),
)

D = torch.nn.Sequential(
    torch.nn.Linear(ART_COMPONENTS,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,1),
    torch.nn.Sigmoid()
)

opt_D = torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(),lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE,N_IDEAS,requires_grad=True)
    G_PAINTINGS = G(G_ideas)
    prob_artist1 = D(G_PAINTINGS)
    G_LOSS = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_LOSS.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_PAINTINGS.detach())
    D_LOSS = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_LOSS.backward(retain_graph=True)
    opt_D.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_PAINTINGS.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()