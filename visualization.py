# demo/visualization.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import pandas as pd
from datetime import datetime

class ModelVisualizer:
    """Класс для визуализации результатов работы модели"""
    
    def __init__(self, inference_model, class_names: Dict[int, str], output_dir: str = 'reports/visualizations'):
        self.model = inference_model
        self.class_names = class_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_predictions(self, image_paths: List[str], true_labels: List[int] = None,
                             save_path: str = None, show: bool = True):
        """Визуализация предсказаний для списка изображений"""
        if true_labels is None:
            true_labels = [None] * len(image_paths)
        
        n_images = len(image_paths)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        for idx, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Загрузка и предсказание
            image = Image.open(img_path).convert('RGB')
            predicted_class, class_name, confidence = self.model.predict(img_path)
            
            # Отображение изображения
            ax.imshow(image)
            
            # Информация о предсказании
            is_correct = (true_label == predicted_class) if true_label is not None else False
            color = 'green' if is_correct else 'red' if true_label is not None else 'blue'
            
            title = f"Pred: {class_name}\nConf: {confidence:.2%}"
            if true_label is not None:
                true_name = self.class_names.get(true_label, f"Class {true_label}")
                title += f"\nTrue: {true_name}"
                title += f"\n{'✓ Correct' if is_correct else '✗ Wrong'}"
            
            ax.set_title(title, fontsize=10, color=color)
            ax.axis('off')
            
            # Рамка вокруг изображения
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                           linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        return fig
    
    def visualize_detailed_predictions(self, image_paths: List[str], true_labels: List[int] = None,
                                      save_path: str = None):
        """Детальная визуализация с дополнительной информацией"""
        if true_labels is None:
            true_labels = [None] * len(image_paths)
        
        n_images = len(image_paths)
        fig = plt.figure(figsize=(15, 5 * n_images))
        
        for idx, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
            gs = GridSpec(3, 4, figure=fig)
            
            # Основное изображение
            ax_img = fig.add_subplot(gs[idx, 0:2])
            image = Image.open(img_path).convert('RGB')
            ax_img.imshow(image)
            ax_img.axis('off')
            
            # Предсказание
            predicted_class, class_name, confidence = self.model.predict(img_path)
            top_k = self.model.get_top_k_predictions(img_path, 5)
            
            # Информация о предсказании
            ax_info = fig.add_subplot(gs[idx, 2:4])
            ax_info.axis('off')
            
            info_text = f"Prediction Details\n"
            info_text += f"{'='*30}\n"
            info_text += f"Predicted: {class_name}\n"
            info_text += f"Confidence: {confidence:.2%}\n"
            if true_label is not None:
                true_name = self.class_names.get(true_label, f"Class {true_label}")
                is_correct = (true_label == predicted_class)
                info_text += f"True: {true_name}\n"
                info_text += f"Status: {'✓ Correct' if is_correct else '✗ Wrong'}\n"
            info_text += f"\nTop-5 Predictions:\n"
            info_text += f"{'-'*20}\n"
            
            for i, (cls_id, cls_name, conf) in enumerate(top_k, 1):
                info_text += f"{i}. {cls_name}: {conf:.2%}\n"
            
            ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed visualization saved to {save_path}")
        
        plt.show()
        plt.close()
        return fig
    
    def visualize_error_analysis(self, image_paths: List[str], true_labels: List[int],
                                save_path: str = None):
        """Визуализация ошибок модели"""
        errors = []
        for img_path, true_label in zip(image_paths, true_labels):
            predicted_class, class_name, confidence = self.model.predict(img_path)
            is_correct = (true_label == predicted_class)
            if not is_correct:
                errors.append({
                    'path': img_path,
                    'true_label': true_label,
                    'true_name': self.class_names.get(true_label, f"Class {true_label}"),
                    'predicted': predicted_class,
                    'predicted_name': class_name,
                    'confidence': confidence
                })
        
        if not errors:
            print("No errors found in the provided images!")
            return None
        
        n_errors = min(len(errors), 10)
        selected_errors = random.sample(errors, n_errors) if len(errors) > n_errors else errors
        
        fig, axes = plt.subplots(2, (n_errors + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_errors > 1 else [axes]
        
        for idx, error in enumerate(selected_errors):
            ax = axes[idx]
            
            image = Image.open(error['path']).convert('RGB')
            ax.imshow(image)
            ax.axis('off')
            
            title = f"True: {error['true_name']}\n"
            title += f"Pred: {error['predicted_name']}\n"
            title += f"Conf: {error['confidence']:.2%}"
            ax.set_title(title, fontsize=10, color='red')
            
            # Красная рамка для ошибок
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # Скрываем лишние подграфики
        for idx in range(len(selected_errors), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Error Analysis - Incorrect Predictions", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error analysis saved to {save_path}")
        
        plt.show()
        plt.close()
        return fig
    
    def create_prediction_grid(self, image_paths: List[str], true_labels: List[int] = None,
                              grid_size: Tuple[int, int] = (2, 5), save_path: str = None):
        """Создание сетки предсказаний с метриками"""
        n_images = grid_size[0] * grid_size[1]
        selected_paths = random.sample(image_paths, min(n_images, len(image_paths)))
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 10))
        axes = axes.flatten()
        
        predictions_data = []
        
        for idx, img_path in enumerate(selected_paths):
            ax = axes[idx]
            
            # Предсказание
            image = Image.open(img_path).convert('RGB')
            predicted_class, class_name, confidence = self.model.predict(img_path)
            top_k = self.model.get_top_k_predictions(img_path, 3)
            
            # Определение правильности
            true_label = None
            if true_labels and idx < len(true_labels):
                true_label = true_labels[idx]
            
            is_correct = (true_label == predicted_class) if true_label is not None else None
            
            # Отображение
            ax.imshow(image)
            ax.axis('off')
            
            # Цвет рамки
            if is_correct is True:
                color = 'green'
                status = '✓'
            elif is_correct is False:
                color = 'red'
                status = '✗'
            else:
                color = 'blue'
                status = '?'
            
            # Заголовок
            title = f"{class_name}\n"
            title += f"Conf: {confidence:.2%}\n"
            if true_label is not None:
                true_name = self.class_names.get(true_label, f"Class {true_label}")
                title += f"True: {true_name}\n"
            title += f"Status: {status}"
            
            ax.set_title(title, fontsize=9, color=color)
            
            # Рамка
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Сохранение данных для статистики
            predictions_data.append({
                'image': os.path.basename(img_path),
                'predicted': predicted_class,
                'predicted_name': class_name,
                'confidence': confidence,
                'true_label': true_label,
                'correct': is_correct,
                'top1': top_k[0][1] if len(top_k) > 0 else '',
                'top1_conf': top_k[0][2] if len(top_k) > 0 else 0
            })
        
        # Скрываем лишние подграфики
        for idx in range(len(selected_paths), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Traffic Sign Recognition - Predictions Grid", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prediction grid saved to {save_path}")
        
        plt.show()
        plt.close()
        
        # Сохранение статистики
        stats_df = pd.DataFrame(predictions_data)
        stats_path = os.path.join(self.output_dir, 'predictions_stats.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Prediction statistics saved to {stats_path}")
        
        return fig, stats_df

    def create_comprehensive_report(self, test_image_paths: List[str], 
                                   true_labels: List[int],
                                   model_name: str = "Current Model"):
        """Создание полного отчета с визуализациями"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"Creating comprehensive report for {model_name}...")
        print(f"Report directory: {report_dir}")
        
        # 1. Случайная выборка 10 изображений
        n_samples = min(10, len(test_image_paths))
        sample_indices = random.sample(range(len(test_image_paths)), n_samples)
        sample_paths = [test_image_paths[i] for i in sample_indices]
        sample_labels = [true_labels[i] for i in sample_indices] if true_labels else None
        
        # 2. Базовая визуализация
        self.visualize_predictions(
            sample_paths, 
            sample_labels,
            save_path=os.path.join(report_dir, 'predictions_overview.png'),
            show=False
        )
        
        # 3. Детальная визуализация
        self.visualize_detailed_predictions(
            sample_paths,
            sample_labels,
            save_path=os.path.join(report_dir, 'detailed_predictions.png')
        )
        
        # 4. Сетка предсказаний
        self.create_prediction_grid(
            sample_paths,
            sample_labels,
            grid_size=(2, 5),
            save_path=os.path.join(report_dir, 'prediction_grid.png')
        )
        
        # 5. Анализ ошибок
        self.visualize_error_analysis(
            sample_paths,
            sample_labels if sample_labels else [],
            save_path=os.path.join(report_dir, 'error_analysis.png')
        )
        
        # 6. Статистика по всем изображениям
        self.analyze_performance(
            test_image_paths,
            true_labels,
            save_path=os.path.join(report_dir, 'performance_stats.png')
        )
        
        print(f"\n✅ Report completed! All files saved to: {report_dir}")
        return report_dir

    def analyze_performance(self, image_paths: List[str], true_labels: List[int],
                           save_path: str = None):
        """Анализ производительности модели на тестовой выборке"""
        predictions = []
        confidences = []
        correct_predictions = 0
        errors = []
        
        for img_path, true_label in zip(image_paths, true_labels):
            pred_class, class_name, confidence = self.model.predict(img_path)
            is_correct = (pred_class == true_label)
            
            predictions.append({
                'path': img_path,
                'true_label': true_label,
                'predicted': pred_class,
                'correct': is_correct,
                'confidence': confidence
            })
            
            confidences.append(confidence)
            if is_correct:
                correct_predictions += 1
            else:
                errors.append((img_path, true_label, pred_class, confidence))
        
        accuracy = correct_predictions / len(image_paths) if image_paths else 0
        
        # Визуализация метрик
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение уверенности
        ax1 = axes[0, 0]
        ax1.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2%}')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.legend()
        
        # 2. Accuracy по классам
        ax2 = axes[0, 1]
        class_predictions = {}
        for pred in predictions:
            true_label = pred['true_label']
            if true_label not in class_predictions:
                class_predictions[true_label] = {'correct': 0, 'total': 0}
            class_predictions[true_label]['total'] += 1
            if pred['correct']:
                class_predictions[true_label]['correct'] += 1
        
        if class_predictions:
            classes = list(class_predictions.keys())
            accuracies = [
                class_predictions[c]['correct'] / class_predictions[c]['total']
                for c in classes
            ]
            # Сортируем по убыванию
            sorted_data = sorted(zip(classes, accuracies), key=lambda x: x[1], reverse=True)
            sorted_classes = [c for c, _ in sorted_data[:15]]  # Топ-15
            sorted_acc = [a for _, a in sorted_data[:15]]
            
            ax2.bar(range(len(sorted_acc)), sorted_acc, alpha=0.7, color='green')
            ax2.set_xticks(range(len(sorted_acc)))
            ax2.set_xticklabels([self.class_names.get(c, f'Class {c}') for c in sorted_classes], 
                               rotation=45, ha='right', fontsize=8)
            ax2.set_ylim([0, 1.1])
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy by Class (Top 15)')
        
        # 3. Общая статистика
        ax3 = axes[1, 0]
        ax3.axis('off')
        stats_text = f"Performance Summary\n"
        stats_text += f"{'='*30}\n"
        stats_text += f"Total Images: {len(image_paths)}\n"
        stats_text += f"Correct: {correct_predictions}\n"
        stats_text += f"Errors: {len(errors)}\n"
        stats_text += f"Accuracy: {accuracy:.2%}\n"
        stats_text += f"Mean Confidence: {np.mean(confidences):.2%}\n"
        stats_text += f"Median Confidence: {np.median(confidences):.2%}\n"
        stats_text += f"Min Confidence: {np.min(confidences):.2%}\n"
        stats_text += f"Max Confidence: {np.max(confidences):.2%}"
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 4. Топ-5 ошибок
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if errors:
            error_text = "Top 5 Errors (Lowest Confidence)\n"
            error_text += f"{'='*30}\n"
            sorted_errors = sorted(errors, key=lambda x: x[3])[:5]
            for idx, (path, true, pred, conf) in enumerate(sorted_errors, 1):
                true_name = self.class_names.get(true, f"Class {true}")
                pred_name = self.class_names.get(pred, f"Class {pred}")
                error_text += f"{idx}. {os.path.basename(path)}\n"
                error_text += f"   True: {true_name}\n"
                error_text += f"   Pred: {pred_name}\n"
                error_text += f"   Conf: {conf:.2%}\n"
        else:
            error_text = "✅ No errors found on this dataset!"
        
        ax4.text(0.05, 0.5, error_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.suptitle("Model Performance Analysis", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance analysis saved to {save_path}")
        
        plt.show()
        plt.close()
        return fig, {'accuracy': accuracy, 'mean_confidence': np.mean(confidences)}

def create_10_images_report(inference_model, test_loader, class_names, output_dir='reports/visualizations'):
    """Создание отчета с 10 изображениями для итогового вывода"""
    # Получение 10 случайных изображений из test_loader
    images = []
    labels = []
    image_paths = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx * test_loader.batch_size >= 10:
            break
        
        # В этом случае data это тензор, нужно получить пути к файлам
        # Это зависит от того, как организован Dataset
        # Для демонстрации создаем фиктивные пути
        for i in range(min(test_loader.batch_size, 10 - len(images))):
            images.append(data[i])
            labels.append(target[i])
            # В реальном приложении здесь должны быть реальные пути к файлам
    
    # Создание визуализации
    visualizer = ModelVisualizer(inference_model, class_names, output_dir)
    
    # Если у нас есть реальные пути к изображениям, используем их
    # Иначе используем тензоры для визуализации
    # Для демонстрации создаем временные файлы из тензоров
    
    temp_dir = 'temp_images'
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_paths = []
    for idx, (img_tensor, label) in enumerate(zip(images, labels)):
        # Сохранение тензора как изображение
        from torchvision.utils import save_image
        temp_path = os.path.join(temp_dir, f'temp_{idx}.jpg')
        save_image(img_tensor, temp_path)
        temp_paths.append(temp_path)
    
    # Создание отчета
    report_dir = visualizer.create_comprehensive_report(
        temp_paths,
        labels,
        model_name="Best Model"
    )
    
    # Очистка временных файлов
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return report_dir