# models/pretrained_models.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from .base_model import BaseModel

class ResNet50Model(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "ResNet50"
        
    def build_model(self):
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except:
            model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, self.num_classes)
        return model
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class EfficientNetB0Model(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "EfficientNetB0"
        
    def build_model(self):
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except:
            model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Linear(1280, self.num_classes)
        return model
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class DenseNet121Model(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "DenseNet121"
        
    def build_model(self):
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except:
            model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, self.num_classes)
        return model
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class MobileNetV3LargeModel(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "MobileNetV3Large"
        
    def build_model(self):
        try:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        except:
            model = models.mobilenet_v3_large(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(960, self.num_classes)
        )
        return model
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class VisionTransformerModel(BaseModel):
    def __init__(self, num_classes, input_size=(224, 224)):
        super().__init__(num_classes, input_size)
        self.name = "ViTBase"
        
    def build_model(self):
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        except:
            model = models.vit_b_16(pretrained=True)
        model.heads = nn.Linear(768, self.num_classes)
        return model
    
    def get_preprocessing(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
