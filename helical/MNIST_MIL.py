import torch 
from torchvision.datasets import MNIST
from torchvision.models import resnet34, feature_extraction
from skimage import io
import numpy as np
from torchsummary import summary

# trainloader
trainloader = MNIST(root = '/Users/marco/mnist', download=True, train=True)
image0, label0 = next(iter(trainloader))
image0 = np.array(image0)
# print(type(image0), type(label0))
# io.imshow(image0)

# model
model = resnet34(weights = 'DEFAULT')
# summary(model, (3, 224, 224) )
names = feature_extraction.get_graph_node_names(model = model)
print(names)
model = feature_extraction.create_feature_extractor(model = model,return_nodes= {'avgpool': 'feat1'})
out = model(torch.rand(1, 3, 1024, 1024))
print([(k, v.shape) for k, v in out.items()])

# create bag and instance labels
