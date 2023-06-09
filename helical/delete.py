import torch 

# tensor = torch.randn(size=[5,4])
tensor = torch.Tensor([[0,1,2,0], [2,1,2,4], [7,3,4,4], [2,3,2,4], [9,7,2,4]])
print(tensor)
print(tensor.size())
print(torch.max(tensor, dim=1))


