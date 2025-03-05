import torch
from torchvision.models import resnet18,ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

#forward pass
prediction = model(data)
loss = (prediction - labels).sum()
#backward pass
loss.backward()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#gradient descent
optimizer.step()
