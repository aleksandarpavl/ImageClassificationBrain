import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import os

# 1. Load model ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 2. Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 3. Load the image
img_path = os.path.join(os.getcwd(), "giraffe.jpg")
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # add batch dimension

# 4. Prediction
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 5. Download ImageNet classes
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes_txt = "imagenet_classes.txt"
if not os.path.exists(classes_txt):
    r = requests.get(url)
    with open(classes_txt, "w") as f:
        f.write(r.text)

with open(classes_txt) as f:
    categories = [line.strip() for line in f.readlines()]

# 6. Top 5 results
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"{categories[top5_catid[i]]} : {top5_prob[i].item() * 100:.2f}%")
