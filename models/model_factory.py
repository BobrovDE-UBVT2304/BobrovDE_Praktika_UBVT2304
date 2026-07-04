# models/model_factory.py
from .cnn_models import SimpleCNN, ResidualCNN
from .pretrained_models import (
    ResNet50Model, EfficientNetB0Model, 
    DenseNet121Model, MobileNetV3LargeModel,
    VisionTransformerModel
)

class ModelFactory:
    _models = {
        'simple_cnn': SimpleCNN,
        'residual_cnn': ResidualCNN,
        'resnet50': ResNet50Model,
        'efficientnet_b0': EfficientNetB0Model,
        'densenet121': DenseNet121Model,
        'mobilenet_v3': MobileNetV3LargeModel,
        'vit_base': VisionTransformerModel,
    }
    
    @classmethod
    def get_available_models(cls):
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, model_name, num_classes, input_size=(224, 224)):
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._models[model_name](num_classes, input_size)
