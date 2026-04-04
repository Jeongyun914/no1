
import torch
from PIL import Image
import torchvision.transforms as transforms
# 2510114 배정윤 뭐라는거야
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28*28, 10)
    def forward(self,x): return self.fc(x.view(-1,28*28))

model = Net()
model.load_state_dict(torch.load('mnist.pth'))
model.eval()

img = Image.open('sample.png').convert('L')
transform = transforms.ToTensor()
x = transform(img)

with torch.no_grad():
    out = model(x)
    pred = out.argmax(1)

print('prediction:', pred.item())
