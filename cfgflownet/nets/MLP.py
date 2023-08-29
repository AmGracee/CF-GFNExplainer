
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,n_inputs, n_hidden,   n_output):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(n_inputs, n_hidden)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(n_hidden, n_output)


    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return x



