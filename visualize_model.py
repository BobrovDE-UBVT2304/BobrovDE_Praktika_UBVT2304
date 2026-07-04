# visualize_model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from torchvision import transforms
from models.model_factory import ModelFactory
import torch.nn.functional as F

class ModelVisualizer:
    def __init__(self, model, model_name, transform, device='cpu'):
        self.model = model
        self.model_name = model_name
        self.transform = transform
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.output_dir = f'reports/model_visualizations_{model_name}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_filters(self):
        """Визуализация фильтров первого сверточного слоя"""
        conv_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                break
        
        if conv_layer is None:
            print("Сверточный слой не найден")
            return
        
        weights = conv_layer.weight.data.cpu().numpy()
        n_filters = min(32, weights.shape[0])
        
        weights = weights[:n_filters]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2))
        if n_rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for idx in range(n_filters):
            filter_img = weights[idx, 0]
            axes[idx].imshow(filter_img, cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(f'F{idx+1}', fontsize=8)
        
        for idx in range(n_filters, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Фильтры сверточного слоя ({self.model_name})', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'filters_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Фильтры сохранены в {save_path}")
        return save_path
    
    def visualize_feature_maps(self, image_path):
        """Визуализация карт признаков"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_value = confidence.item()
        
        # Получаем карты признаков
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer:
            hook = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = self.model(image_tensor.unsqueeze(0))
            hook.remove()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        ax = axes[0]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Оригинал\nКласс: {predicted_class}\nУверенность: {confidence_value:.2%}', fontsize=10)
        
        if feature_maps:
            features = feature_maps[0][0]
            n_channels = min(7, features.shape[0])
            
            for i in range(n_channels):
                ax = axes[i + 1]
                feature_map = features[i].cpu().numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                ax.imshow(feature_map, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Канал {i+1}', fontsize=8)
        
        for idx in range(n_channels + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Карты признаков для {os.path.basename(image_path)}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'feature_maps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Карты признаков сохранены в {save_path}")
        return save_path
    
    def visualize_activations(self, image_path):
        """Визуализация активаций нейронов"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).to(self.device)
        
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d)):
                target_layer = module
                break
        
        if target_layer:
            hook = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = self.model(image_tensor.unsqueeze(0))
            hook.remove()
        
        if activations:
            act = activations[0].cpu().numpy().flatten()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            ax1.hist(act[:1000], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Значение активации')
            ax1.set_ylabel('Частота')
            ax1.set_title('Распределение активаций нейронов')
            ax1.axvline(np.mean(act[:1000]), color='red', linestyle='--', 
                       label=f'Среднее: {np.mean(act[:1000]):.3f}')
            ax1.legend()
            
            ax2 = axes[1]
            top_indices = np.argsort(act)[-20:]
            top_values = act[top_indices]
            ax2.barh(range(len(top_values)), top_values, color='green', edgecolor='darkgreen')
            ax2.set_xlabel('Значение активации')
            ax2.set_ylabel('Нейрон')
            ax2.set_title('Топ-20 активированных нейронов')
            
            plt.suptitle('Активации нейронов модели', fontsize=14)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'activations.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Активации сохранены в {save_path}")
            return save_path
    
    def visualize_gradcam(self, image_path):
        """Упрощенная Grad-CAM визуализация для EfficientNet"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)
            
            # Находим последний сверточный слой
            target_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and 'blocks' in name:
                    target_layer = module
            
            if target_layer is None:
                # Если не нашли, берем первый сверточный слой
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                        break
            
            if target_layer is None:
                print("Сверточный слой не найден для Grad-CAM")
                return
            
            # Получаем активации и градиенты
            activations = []
            gradients = []
            
            def forward_hook(module, input, output):
                activations.append(output)
            
            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])
            
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_backward_hook(backward_hook)
            
            # Forward pass
            output = self.model(image_tensor.unsqueeze(0))
            pred_class = output.argmax(dim=1)
            
            # Backward pass
            self.model.zero_grad()
            output[0, pred_class].backward()
            
            forward_handle.remove()
            backward_handle.remove()
            
            if activations and gradients:
                acts = activations[0][0]
                grads = gradients[0]
                
                # Вычисляем веса каналов
                weights = grads.mean(dim=[1, 2], keepdim=True)
                cam = (weights * acts).sum(dim=0, keepdim=True)
                cam = torch.relu(cam)
                cam = cam.detach().cpu().numpy()[0]
                
                # Нормализация
                if cam.max() > cam.min():
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                else:
                    cam = np.zeros_like(cam)
                
                # Масштабируем до размера изображения
                cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
                cam_resized = cam_resized.resize(image.size, Image.Resampling.BILINEAR)
                cam_resized = np.array(cam_resized) / 255.0
                
                # Визуализация
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image)
                axes[0].axis('off')
                axes[0].set_title('Оригинал')
                
                im = axes[1].imshow(cam_resized, cmap='jet')
                axes[1].axis('off')
                axes[1].set_title('Grad-CAM')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                axes[2].imshow(image)
                axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
                axes[2].axis('off')
                axes[2].set_title(f'Наложение\nКласс: {pred_class.item()}')
                
                plt.suptitle('Grad-CAM: На что смотрит модель', fontsize=14)
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, 'gradcam.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✅ Grad-CAM сохранен в {save_path}")
                return save_path
                
        except Exception as e:
            print(f"Grad-CAM пропущен: {e}")
            return None
    
    def visualize_predictions_comparison(self, image_paths):
        """Сравнение предсказаний на нескольких изображениях"""
        n_images = len(image_paths)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        # Преобразуем axes в плоский список
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, img_path in enumerate(image_paths):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            ax.imshow(image)
            ax.axis('off')
            
            pred_class = predicted.item()
            conf = confidence.item()
            
            # Определяем класс из имени файла
            true_class = None
            try:
                # Имя файла содержит класс в начале (например 00000_00000.png)
                true_class = int(os.path.basename(img_path).split('_')[0])
            except:
                pass
            
            if true_class is not None:
                is_correct = (pred_class == true_class)
                color = 'green' if is_correct else 'red'
                status = '✓ Верно' if is_correct else '✗ Ошибка'
                title = f'Класс: {pred_class}\nУверенность: {conf:.1%}\n{status}'
            else:
                color = 'blue'
                title = f'Класс: {pred_class}\nУверенность: {conf:.1%}'
            
            ax.set_title(title, fontsize=9, color=color)
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Скрываем лишние подграфики
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Результаты предсказаний ({self.model_name})', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'predictions_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Сравнение предсказаний сохранено в {save_path}")
        return save_path
    
    def create_comprehensive_report(self, image_path):
        """Создание полного отчета визуализации"""
        print(f"\n📊 Создание визуализаций для {os.path.basename(image_path)}")
        print("="*60)
        
        self.visualize_filters()
        self.visualize_feature_maps(image_path)
        self.visualize_activations(image_path)
        self.visualize_gradcam(image_path)
        
        print(f"\n✅ Все визуализации сохранены в {self.output_dir}")
        return self.output_dir

def load_best_model():
    """Загрузка лучшей обученной модели"""
    if not os.path.exists('reports/results.json'):
        print("❌ Результаты не найдены! Сначала запустите обучение.")
        return None, None, None
    
    with open('reports/results.json', 'r') as f:
        results = json.load(f)
    
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"🏆 Лучшая модель: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    model_interface = ModelFactory.create_model(best_model_name, 43, (128, 128))
    model = model_interface.get_model()
    
    model_path = f"checkpoints/{best_model_name}_final.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Модель загружена из {model_path}")
        return model, model_interface.get_preprocessing(), best_model_name
    else:
        print(f"❌ Файл {model_path} не найден!")
        return None, None, None

def main():
    model, transform, model_name = load_best_model()
    
    if model is None:
        print("❌ Не удалось загрузить модель!")
        return
    
    visualizer = ModelVisualizer(model, model_name, transform)
    
    print("\n" + "="*60)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ МОДЕЛИ")
    print("="*60)
    
    # 1. Фильтры
    print("\n1. Визуализация фильтров...")
    visualizer.visualize_filters()
    
    # 2. Тестовые изображения
    test_dir = 'data/processed/gtsrb'
    test_images = []
    
    if os.path.exists(test_dir):
        for class_dir in os.listdir(test_dir)[:10]:
            class_path = os.path.join(test_dir, class_dir)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.endswith(('.ppm', '.jpg', '.png'))]
                if images:
                    test_images.append(os.path.join(class_path, images[0]))
    
    if test_images:
        print(f"\n2. Найдено {len(test_images)} тестовых изображений")
        for img_path in test_images[:3]:
            print(f"\nОбработка: {os.path.basename(img_path)}")
            visualizer.visualize_feature_maps(img_path)
            visualizer.visualize_activations(img_path)
            visualizer.visualize_gradcam(img_path)
        
        if len(test_images) >= 5:
            print("\n3. Сравнение предсказаний на 5 изображениях...")
            visualizer.visualize_predictions_comparison(test_images[:5])
    
    print(f"\n✅ Все визуализации завершены!")
    print(f"📁 Результаты в папке: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
