# demo/inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, Tuple, List
import json

class TrafficSignInference:
    """Класс для инференса модели распознавания дорожных знаков"""
    
    def __init__(self, model_path: str, model_interface, num_classes: int = 43,
                 device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_interface.get_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = model_interface.get_preprocessing()
        self.num_classes = num_classes
        
        # Загрузка названий классов
        self.class_names = self._load_class_names()
    
    def _load_class_names(self) -> Dict[int, str]:
        """Загрузка названий классов"""
        # Загрузка из файла или использование стандартных
        try:
            with open('data/class_names.json', 'r') as f:
                names = json.load(f)
                return {int(k): v for k, v in names.items()}
        except:
            # Стандартные классы GTSRB
            names = {
                0: 'Speed limit (20km/h)',
                1: 'Speed limit (30km/h)',
                2: 'Speed limit (50km/h)',
                3: 'Speed limit (60km/h)',
                4: 'Speed limit (70km/h)',
                5: 'Speed limit (80km/h)',
                6: 'End of speed limit (80km/h)',
                7: 'Speed limit (100km/h)',
                8: 'Speed limit (120km/h)',
                9: 'No passing',
                10: 'No passing for trucks',
                11: 'Right-of-way at intersection',
                12: 'Priority road',
                13: 'Yield',
                14: 'Stop',
                15: 'No vehicles',
                16: 'Vehicles over 3.5t prohibited',
                17: 'No entry',
                18: 'General caution',
                19: 'Dangerous curve left',
                20: 'Dangerous curve right',
                21: 'Double curve',
                22: 'Bumpy road',
                23: 'Slippery road',
                24: 'Road narrows on right',
                25: 'Road work',
                26: 'Traffic signals',
                27: 'Pedestrians',
                28: 'Children crossing',
                29: 'Bicycles crossing',
                30: 'Beware of ice/snow',
                31: 'Wild animals crossing',
                32: 'End of speed limit + overtaking',
                33: 'Turn right ahead',
                34: 'Turn left ahead',
                35: 'Ahead only',
                36: 'Go straight or right',
                37: 'Go straight or left',
                38: 'Keep right',
                39: 'Keep left',
                40: 'Roundabout',
                41: 'End of no passing',
                42: 'End of no passing (trucks)'
            }
            return names
    
    def predict(self, image_path: str) -> Tuple[int, str, float]:
        """Предсказание класса для изображения"""
        # Загрузка и предобработка
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        class_name = self.class_names.get(predicted_class, f"Class {predicted_class}")
        
        return predicted_class, class_name, confidence_score
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, str, float]]:
        """Предсказание для пакета изображений"""
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results
    
    def get_top_k_predictions(self, image_path: str, k: int = 5) -> List[Tuple[int, str, float]]:
        """Получение топ-K предсказаний"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_k_values, top_k_indices = torch.topk(probabilities, k, dim=1)
        
        results = []
        for i in range(k):
            class_idx = top_k_indices[0][i].item()
            confidence = top_k_values[0][i].item()
            class_name = self.class_names.get(class_idx, f"Class {class_idx}")
            results.append((class_idx, class_name, confidence))
        
        return results
    
    def analyze_errors(self, image_path: str, true_class: int) -> Dict:
        """Анализ ошибок для изображения"""
        predicted_class, class_name, confidence = self.predict(image_path)
        
        is_correct = (predicted_class == true_class)
        top_k = self.get_top_k_predictions(image_path, 3)
        
        return {
            'true_class': true_class,
            'true_class_name': self.class_names.get(true_class, f"Class {true_class}"),
            'predicted_class': predicted_class,
            'predicted_class_name': class_name,
            'confidence': confidence,
            'is_correct': is_correct,
            'top_3_predictions': top_k
        }