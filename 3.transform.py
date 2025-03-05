#데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공 안됨
#학습에 필요하도록 변형하는게 Transform
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
