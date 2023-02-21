import torch 
from MIL_aggregator import LogisticRegression, NN
from MIL_aggregator2 import AttentionSoftMax


class MIL_NN(torch.nn.Module):
    def __init__(self, n=512,  
                 n_mid=1024, 
                 n_classes=1, 
                 dropout=0.1,
                 agg = None,
                 scoring=None,
                ):
        super(MIL_NN, self).__init__()
        self.agg = agg if agg is not None else AttentionSoftMax(n)
        
        if n_mid == 0:
            self.bag_model = LogisticRegression(n, n_classes)
        else:
            self.bag_model = NN(n, n_mid, n_classes, dropout=dropout, scoring=scoring)
        
    def forward(self, bag_features, bag_lbls=None):
        """
        bag_feature is an aggregated vector of 512 features
        bag_att is a gate vector of n_inst instances
        bag_lbl is a vector a labels
        figure out batches
        """
        bag_feature, bag_att, bag_keys = list(zip(*[list(self.agg(ff.float())) + [idx]
                                                    for idx, ff in (bag_features.items())]))
        bag_att = dict(zip(bag_keys, [a.detach().cpu() for a  in bag_att]))
        bag_feature_stacked = torch.stack(bag_feature)
        y_pred = self.bag_model(bag_feature_stacked)
        return y_pred, bag_att, bag_keys