import numpy as np

from metadrive.utils.math_utils import not_zero, wrap_to_pi
from model import Resnet, Resnet_Categorize
import torch
from torchvision import transforms

from PIL import Image

# model = Resnet_Categorize(mode='linear',pretrained=True)
# checkpoint = torch.load("model_categorize.pt", map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

model = Resnet_Categorize()
model.load_state_dict(torch.load("model_categorize_revised.pt", map_location=torch.device('cpu')))
model.eval()

image_path = "./dataset/val/(0.10624197352048775, -0.0027929785415545805, 29.14923399838864).png"

image = Image.open(image_path)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((96,96)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
image = data_transform(image)

image = torch.unsqueeze(image, 0)

res = model(image)

print(res)