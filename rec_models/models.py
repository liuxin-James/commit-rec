import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class RecNet(nn.Module):
    def __init__(self, n_features) -> None:
        super(RecNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        logits = self.model(x)
        logits = F.softmax(logits, dim=1)
        outputs = (logits,)
        if y is not None:
            y_ = y.squeeze(1)
            loss_value = self.loss(logits, y_)
            outputs = (loss_value,)+outputs
        return outputs


class WideComponent(nn.Module):
    def __init__(self, n_features):
        super(WideComponent, self).__init__()
        self.linear = nn.Linear(n_features=n_features, out_features=1)

    def forward(self, x):
        return self.linear(x)


class DnnComponent(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DnnComponent, self).__init__()

        self.dnn = nn.ModuleList([nn.Linear(layer[0], layer[1])for layer in list(
            zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        for linear in self.dnn:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)

        return x


class DQN:
    def __init__(self, model: nn.Module, memory, cfg) -> None:
        self.n_actions = cfg["n_actions"]
        self.gamma = cfg["gamma"]
        self.sample_count = 0
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_start = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay = cfg["epsilong_decay"]
        self.batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.memory = memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg["lr"])

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy(param.data)

    def sample_action(self, state):
        self.sample_count += 1

        self.epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)

        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(
                    state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device,
                             dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item()
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)

        state_batch = torch.tensor(
            np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(
            action_batch, device=self.device).unsqueeze(dim=1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(
            np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1,index=action_batch)
        nex_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * nex_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values,expected_q_values.unsequeeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()