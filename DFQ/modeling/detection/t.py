import torch


state_dict1 = torch.load('SkyNet.pth')
state_dict2 = torch.load('SkyNet_1.pth')

for np, p in state_dict1.items():
    print('-' * 10)
    print(np)
    print(state_dict1[np] == state_dict2[np])
    print()
print()