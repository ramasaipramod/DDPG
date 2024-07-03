import torch
import torch.nn as nn 

class critic(nn.Module):
    def __init__(self, nstate, naction):
        super(critic, self).__init__()
        self.layer1 = nn.Linear(nstate+naction, 120)
        self.layer2 = nn.Linear(120, 60)
        self.layer3 = nn.Linear(60, 20)
        self.layer4 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        act = self.relu(self.layer1(input.to(torch.float32)))
        act = self.relu(self.layer2(act))
        act = self.relu(self.layer3(act))
        act = self.relu(self.layer4(act))
        return act

class actor(nn.Module):
    def __init__(self, nstate):
        super(actor, self).__init__()
        self.layer1 = nn.Linear(nstate, 150)
        self.layer2 = nn.Linear(150, 70)
        self.layer3 = nn.Linear(70, 17)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        q = self.relu(self.layer1(input.to(torch.float32)))
        q = self.relu(self.layer2(q))
        q = self.layer3(q)
        return 0.4*self.tanh(q)