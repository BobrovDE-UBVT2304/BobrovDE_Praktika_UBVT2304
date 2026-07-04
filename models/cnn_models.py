# models/cnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from torchvision import transforms

class SimpleCNN(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "SimpleCNN"
        
    def build_model(self):
        class CNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, num_classes)
                
            def forward(self, x):
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        return CNN(self.num_classes)
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class ResidualCNN(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "ResidualCNN"
        
    def build_model(self):
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
                
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return F.relu(out)
                
        class ResCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.layer1 = ResidualBlock(64, 64)
                self.layer2 = ResidualBlock(64, 128)
                self.layer3 = ResidualBlock(128, 256)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, num_classes)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = F.max_pool2d(x, 2)
                x = self.layer2(x)
                x = F.max_pool2d(x, 2)
                x = self.layer3(x)
                x = F.max_pool2d(x, 2)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        return ResCNN(self.num_classes)
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
