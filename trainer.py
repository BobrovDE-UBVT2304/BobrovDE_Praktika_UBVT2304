# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
import json
import time
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image

class Trainer:
    """Класс для обучения и оценки моделей"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_time': []
        }
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs, lr=0.001, weight_decay=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Валидация
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            epoch_time = time.time() - epoch_start
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_time'].append(epoch_time)
            
            # Обновление learning rate
            scheduler.step(val_loss)
            
            # Сохранение лучшей модели
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"{self.config.save_dir}/best_model_{self.model.__class__.__name__}.pth")
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            print("-" * 50)
        
        return self.history
    
    def test(self, test_loader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc, all_preds, all_labels = self.validate(test_loader, criterion)
        
        # Вычисление дополнительных метрик
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Model loaded from {path}")