
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
#----------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import numpy as np

#Define the model architecture and load the pre-trained model 'Lenet'
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# device = "cuda" if torch.cuda.is_available else "cpu"
device = 'cpu'
print(device)

def load_model():
    Lenet = LeNet().to(device)
    model_path = os.getcwd() +"/lenet_MNIST.pth"
    print(model_path)
    Lenet = torch.load(model_path, map_location=torch.device(device))
    return Lenet

def predict_number(image):
    image_tensor = preprocess_image(image)
    model = load_model()
    output = model(image_tensor)
    prediction = torch.max(output,1)[1]
    return prediction.item()


def preprocess_image(image):
    # Read the image in grayscale
    if image is not None:
        # Convert the uploaded image to a numpy array
        image_array = np.fromstring(image.read(), np.uint8)
    
        # Decode the image using OpenCV
        processed_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 32x32 pixels
    processed_image = cv2.resize(processed_image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    processed_image = processed_image / 255.0

    image_tensor = torch.tensor(processed_image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
        
    return image_tensor
