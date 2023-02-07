import torch 
from torch import nn


class NoisyAnd(torch.nn.Module):
    def __init__(self, a=10, dims=[1,2]):
        super(NoisyAnd, self).__init__()
#         self.output_dim = output_dim
        self.a = a
        self.b = torch.nn.Parameter(torch.tensor(0.01))
        self.dims =dims
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
#         h_relu = self.linear1(x).clamp(min=0)
        mean = torch.mean(x, self.dims, True)
        res = (self.sigmoid(self.a * (mean - self.b)) - self.sigmoid(-self.a * self.b)) / (
              self.sigmoid(self.a * (1 - self.b)) - self.sigmoid(-self.a * self.b))
        return res
    


class NN(torch.nn.Module):

    def __init__(self, n=512, n_mid = 1024,
                 n_out=1, dropout=0.2,
                 scoring = None,
                ):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(n, n_mid)
        self.non_linearity = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(n_mid, n_out)
        self.dropout = torch.nn.Dropout(dropout)
        if scoring:
            self.scoring = scoring
        else:
            self.scoring = torch.nn.Softmax() if n_out>1 else torch.nn.Sigmoid()
        
    def forward(self, x):
        z = self.linear1(x)
        z = self.non_linearity(z)
        z = self.dropout(z)
        z = self.linear2(z)
        y_pred = self.scoring(z)
        return y_pred
    

class LogisticRegression(torch.nn.Module):
    def __init__(self, n=512, n_out=1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n, n_out)
        self.scoring = torch.nn.Softmax() if n_out>1 else torch.nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        y_pred = self.scoring(z)
        return y_pred

    
def regularization_loss(params,
                        reg_factor = 0.005,
                        reg_alpha = 0.5):
    params = [pp for pp in params if len(pp.shape)>1]
    l1_reg = nn.L1Loss()
    l2_reg = nn.MSELoss()
    loss_reg =0
    for pp in params:
        loss_reg+=reg_factor*((1-reg_alpha)*l1_reg(pp, target=torch.zeros_like(pp)) +\
                           reg_alpha*l2_reg(pp, target=torch.zeros_like(pp)))
    return loss_reg