import torch 
from torch import nn


class SoftMaxMeanSimple(torch.nn.Module):
    def __init__(self, n, n_inst, dim=0):
        """
        if dim==1:
            given a tensor `x` with dimensions [N * M],
            where M -- dimensionality of the featur vector
                       (number of features per instance)
                  N -- number of instances
            initialize with `AggModule(M)`
            returns:
            - weighted result: [M]
            - gate: [N]
        if dim==0:
            ...
        """
        super(SoftMaxMeanSimple, self).__init__()
        self.dim = dim
        self.gate = torch.nn.Softmax(dim=self.dim)      
        self.mdl_instance_transform = nn.Sequential(
                            nn.Linear(n, n_inst),
                            nn.LeakyReLU(),
                            nn.Linear(n_inst, n),
                            nn.LeakyReLU(),
                            )
    def forward(self, x):
        z = self.mdl_instance_transform(x)
        if self.dim==0:
            z = z.view((z.shape[0],1)).sum(1)
        elif self.dim==1:
            z = z.view((1, z.shape[1])).sum(0)
        gate_ = self.gate(z)
        res = torch.sum(x* gate_, self.dim)
        return res, gate_

    
class AttentionSoftMax(torch.nn.Module):
    def __init__(self, in_features = 3, out_features = None):
        """
        given a tensor `x` with dimensions [N * M],
        where M -- dimensionality of the featur vector
                   (number of features per instance)
              N -- number of instances
        initialize with `AggModule(M)`
        returns:
        - weighted result: [M]
        - gate: [N]
        """
        super(AttentionSoftMax, self).__init__()
        self.otherdim = ''
        if out_features is None:
            out_features = in_features
        # print(f"number in features: {in_features}")
        self.layer_linear_tr = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_linear_query = nn.Linear(out_features, 1)
        # print('9')
        
    def forward(self, x):
        # print('bug here?1')
        # print(x.shape)
        keys = self.layer_linear_tr(x)
        # print('bug here?2')
        keys = self.activation(keys)
        # print(f'{keys.shape}')
        attention_map_raw = self.layer_linear_query(keys)[...,0]
        # print(f'{attention_map_raw.shape}')
        # print('bug here?4')
        attention_map = nn.Softmax(dim=-1)(attention_map_raw)
        # print('bug here?4')

        # print(attention_map.shape)
        result = torch.einsum(f'{self.otherdim}i,{self.otherdim}ij->{self.otherdim}j', attention_map, x)
        return result, attention_map