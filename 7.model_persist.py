import torch
import torchvision.models as models

#torch save를 이용하여 학습한 매개변수 state_dict를 저장
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), ' model_weights.pth')

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

#모델의 형태를 저장, 불러오기
torch.save(model, 'model.pth')
model = torch.load('model.pth')
