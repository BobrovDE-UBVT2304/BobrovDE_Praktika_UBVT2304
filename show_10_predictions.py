# show_10_predictions.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import random
from torchvision import transforms
from models.model_factory import ModelFactory
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def load_best_model():
    """Загрузка лучшей обученной модели"""
    if not os.path.exists('reports/results.json'):
        print("❌ Результаты не найдены!")
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

def get_test_images(test_dir, num_images=10):
    """Получение случайных тестовых изображений"""
    test_images = []
    test_labels = []
    
    if not os.path.exists(test_dir):
        print(f"❌ Папка {test_dir} не найдена!")
        return [], []
    
    all_images = []
    all_labels = []
    
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            try:
                label = int(class_dir)
            except:
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.endswith(('.ppm', '.jpg', '.png'))]
            
            for img in images:
                all_images.append(os.path.join(class_path, img))
                all_labels.append(label)
    
    if len(all_images) == 0:
        print("❌ Нет изображений для тестирования!")
        return [], []
    
    if len(all_images) > num_images:
        indices = random.sample(range(len(all_images)), num_images)
        selected_images = [all_images[i] for i in indices]
        selected_labels = [all_labels[i] for i in indices]
    else:
        selected_images = all_images
        selected_labels = all_labels
    
    print(f"✅ Выбрано {len(selected_images)} изображений")
    return selected_images, selected_labels

def get_feature_maps(model, transform, image_path, device='cpu'):
    """Получение карт признаков из первого сверточного слоя"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to(device)
    
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Находим первый сверточный слой
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    if target_layer:
        hook = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(image_tensor.unsqueeze(0))
        hook.remove()
    
    if feature_maps:
        return feature_maps[0][0].cpu().numpy(), image
    return None, image

def predict_image(model, transform, image_path, device='cpu'):
    """Предсказание для одного изображения"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to(device)
    
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5, dim=1)
    
    return top_indices[0].cpu().numpy(), top_probs[0].cpu().numpy(), image

def show_10_predictions_with_features():
    """Показ 10 изображений с картами признаков"""
    
    # Загружаем модель
    model, transform, model_name = load_best_model()
    if model is None:
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Получаем тестовые изображения
    test_dir = 'data/processed/gtsrb'
    images, labels = get_test_images(test_dir, num_images=10)
    
    if not images:
        print("❌ Нет изображений для отображения!")
        return
    
    # Создаем сетку 5x4 (5 строк, 4 колонки)
    fig = plt.figure(figsize=(25, 30))
    gs = gridspec.GridSpec(10, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    correct = 0
    results = []
    
    for idx, (img_path, true_label) in enumerate(zip(images, labels)):
        if idx >= 10:
            break
        
        # Получаем предсказания и карты признаков
        top_classes, top_probs, image = predict_image(model, transform, img_path, device)
        pred_class = top_classes[0]
        confidence = top_probs[0]
        is_correct = (pred_class == true_label)
        
        if is_correct:
            correct += 1
        
        # Получаем карты признаков
        feature_maps, _ = get_feature_maps(model, transform, img_path, device)
        
        results.append({
            'image': os.path.basename(img_path),
            'true_label': true_label,
            'pred_class': pred_class,
            'confidence': confidence,
            'is_correct': is_correct,
            'top_classes': top_classes,
            'top_probs': top_probs
        })
        
        # ---- ОРИГИНАЛЬНОЕ ИЗОБРАЖЕНИЕ (колонка 0) ----
        ax_img = plt.subplot(gs[idx, 0])
        ax_img.imshow(image)
        ax_img.axis('off')
        
        color = '#2ecc71' if is_correct else '#e74c3c'
        
        # Заголовок
        if is_correct:
            title = f'#{idx+1} ✅\nКласс {pred_class}'
        else:
            title = f'#{idx+1} ❌\n{pred_class} (должен {true_label})'
        ax_img.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Рамка
        rect = Rectangle((0, 0), 1, 1, transform=ax_img.transAxes,
                        linewidth=3, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)
        
        # Уверенность
        ax_img.text(0.5, -0.08, f'Уверенность: {confidence:.1%}', 
                   transform=ax_img.transAxes, fontsize=9, ha='center')
        
        # ---- КАРТЫ ПРИЗНАКОВ (колонки 1, 2, 3) ----
        if feature_maps is not None:
            n_features = min(6, feature_maps.shape[0])
            # Показываем первые 6 карт признаков в 3 колонках
            for f_idx in range(min(3, n_features)):
                ax_feat = plt.subplot(gs[idx, f_idx + 1])
                
                feat_map = feature_maps[f_idx]
                # Нормализация
                if feat_map.max() > feat_map.min():
                    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                else:
                    feat_map = np.zeros_like(feat_map)
                
                ax_feat.imshow(feat_map, cmap='viridis')
                ax_feat.axis('off')
                ax_feat.set_title(f'Канал {f_idx+1}', fontsize=8)
    
    # ---- ИТОГОВАЯ СТАТИСТИКА ----
    accuracy = correct / len(images) if images else 0
    
    # Добавляем информационную панель внизу
    fig.text(0.5, 0.02, 
             f'Модель: {model_name} | Точность на 10 изображениях: {accuracy:.1%} ({correct}/{len(images)}) | '
             f'✅ Правильные: {correct} | ❌ Ошибки: {len(images)-correct}',
             fontsize=14, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Заголовок
    fig.suptitle('Результаты предсказаний и карты признаков', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Сохраняем
    save_path = 'reports/10_predictions_with_features.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Результат с картами признаков сохранен в {save_path}")
    
    plt.show()
    plt.close()
    
    # ---- ВЫВОД СТАТИСТИКИ ----
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО 10 ИЗОБРАЖЕНИЯМ")
    print("="*60)
    for i, r in enumerate(results, 1):
        status = "✅" if r['is_correct'] else "❌"
        print(f"{i}. {status} {r['image']}")
        print(f"   Истинный класс: {r['true_label']} -> Предсказано: {r['pred_class']}")
        print(f"   Уверенность: {r['confidence']:.1%}")
        print(f"   Top-5: {list(zip(r['top_classes'], [f'{p:.1%}' for p in r['top_probs']]))}")
        print()
    
    print(f"Точность: {accuracy:.1%} ({correct}/{len(images)})")
    print("="*60)
    
    return save_path, results

def create_feature_maps_grid(image_path, model, transform, device='cpu', n_cols=8):
    """Создание сетки карт признаков для одного изображения"""
    feature_maps, image = get_feature_maps(model, transform, image_path, device)
    
    if feature_maps is None:
        return None
    
    n_features = min(32, feature_maps.shape[0])
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.5))
    if n_rows == 1:
        axes = [axes]
    axes = axes.flatten()
    
    # Показываем оригинальное изображение
    ax = axes[0]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Оригинал', fontsize=10, fontweight='bold')
    
    # Показываем карты признаков
    for idx in range(n_features):
        ax = axes[idx + 1] if idx + 1 < len(axes) else None
        if ax:
            feat_map = feature_maps[idx]
            if feat_map.max() > feat_map.min():
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            else:
                feat_map = np.zeros_like(feat_map)
            ax.imshow(feat_map, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'F{idx+1}', fontsize=8)
    
    # Скрываем лишние
    for idx in range(n_features + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Карты признаков для {os.path.basename(image_path)}', fontsize=14)
    plt.tight_layout()
    
    save_path = f'reports/feature_maps_{os.path.basename(image_path).split(".")[0]}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Карты признаков сохранены в {save_path}")
    return save_path

def show_detailed_10_predictions():
    """Показ 10 изображений с детальной информацией"""
    
    model, transform, model_name = load_best_model()
    if model is None:
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    test_dir = 'data/processed/gtsrb'
    images, labels = get_test_images(test_dir, num_images=10)
    
    if not images:
        return
    
    # Создаем большую сетку
    fig = plt.figure(figsize=(25, 15))
    
    for idx, (img_path, true_label) in enumerate(zip(images, labels)):
        if idx >= 10:
            break
        
        # Создаем подсетку для каждого изображения (3x3)
        gs = gridspec.GridSpecFromSubplotSpec(3, 3, 
                                             subplot_spec=plt.subplot(2, 5, idx+1),
                                             hspace=0.1, wspace=0.1)
        
        # Получаем данные
        top_classes, top_probs, image = predict_image(model, transform, img_path, device)
        pred_class = top_classes[0]
        confidence = top_probs[0]
        is_correct = (pred_class == true_label)
        
        feature_maps, _ = get_feature_maps(model, transform, img_path, device)
        
        # ---- ОРИГИНАЛ (центр) ----
        ax_center = plt.subplot(gs[1, 1])
        ax_center.imshow(image)
        ax_center.axis('off')
        
        color = '#2ecc71' if is_correct else '#e74c3c'
        rect = Rectangle((0, 0), 1, 1, transform=ax_center.transAxes,
                        linewidth=2, edgecolor=color, facecolor='none')
        ax_center.add_patch(rect)
        
        # ---- КАРТЫ ПРИЗНАКОВ (вокруг) ----
        if feature_maps is not None:
            # Верхний ряд
            for i, pos in enumerate([(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]):
                if i >= 8:
                    break
                if i < feature_maps.shape[0]:
                    ax = plt.subplot(gs[pos[0], pos[1]])
                    feat_map = feature_maps[i]
                    if feat_map.max() > feat_map.min():
                        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                    else:
                        feat_map = np.zeros_like(feat_map)
                    ax.imshow(feat_map, cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'F{i+1}', fontsize=6)
        
        # ---- ИНФОРМАЦИЯ (под изображением) ----
        ax_info = plt.subplot(gs[2, 1])
        ax_info.axis('off')
        
        info_text = f"#{idx+1}\n"
        info_text += f"Истинный: {true_label}\n"
        info_text += f"Предсказано: {pred_class}\n"
        info_text += f"Уверенность: {confidence:.1%}\n"
        info_text += f"{'✅ Верно' if is_correct else '❌ Ошибка'}"
        
        ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Детальный анализ 10 изображений с картами признаков\nМодель: {model_name}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    save_path = 'reports/detailed_10_predictions_with_features.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Детальный отчет сохранен в {save_path}")
    plt.show()
    plt.close()
    
    return save_path

def create_complete_report():
    """Создание полного отчета"""
    print("\n" + "="*60)
    print("СОЗДАНИЕ ПОЛНОГО ОТЧЕТА С КАРТАМИ ПРИЗНАКОВ")
    print("="*60)
    
    # 1. Основной отчет с 10 изображениями и картами признаков
    print("\n1. Создание основного отчета...")
    save_path, results = show_10_predictions_with_features()
    
    # 2. Детальный отчет
    print("\n2. Создание детального отчета...")
    save_path_detailed = show_detailed_10_predictions()
    
    # 3. Отдельные карты признаков для нескольких изображений
    print("\n3. Создание отдельных карт признаков...")
    model, transform, model_name = load_best_model()
    if model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        test_dir = 'data/processed/gtsrb'
        images, _ = get_test_images(test_dir, num_images=3)
        
        for img_path in images[:3]:
            create_feature_maps_grid(img_path, model, transform, device)
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"📁 Основной отчет: {save_path}")
    print(f"📁 Детальный отчет: {save_path_detailed}")
    print(f"📁 Отдельные карты признаков в папке reports/")
    print("\nВсе файлы сохранены в папке reports/")

if __name__ == "__main__":
    create_complete_report()
