# training/metrics.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from typing import Dict, List, Tuple, Any

class ModelEvaluator:
    """Класс для оценки и сравнения моделей"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, test_loader, model_name: str) -> Dict[str, Any]:
        """Оценка одной модели"""
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                
                # Измерение времени инференса
                start_time = time.time()
                output = model(data)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())
        
        # Вычисление метрик
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Размер модели
        model_size = sum(p.numel() for p in model.parameters()) / 1e6  # в миллионах параметров
        
        # Время инференса
        avg_inference_time = np.mean(inference_times) * 1000  # в миллисекундах
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model_size_mb': model_size,
            'avg_inference_time_ms': avg_inference_time,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models_dict: Dict[str, torch.nn.Module], 
                       test_loader) -> Dict[str, Dict]:
        """Сравнение нескольких моделей"""
        results = {}
        
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, test_loader, name)
        
        return results
    
    def generate_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Генерация таблицы сравнения"""
        import pandas as pd
        
        df_data = []
        for model_name, metrics in results.items():
            df_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'Model Size (M)': metrics['model_size_mb'],
                'Inference (ms)': metrics['avg_inference_time_ms']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Accuracy', ascending=False)
        return df
    
    def plot_confusion_matrix(self, results: Dict[str, Dict], model_name: str, 
                              class_names: List[str] = None):
        """Построение confusion matrix"""
        data = results[model_name]
        cm = confusion_matrix(data['labels'], data['predictions'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=class_names[:20] if class_names else 'auto',
                   yticklabels=class_names[:20] if class_names else 'auto')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    def plot_comparison(self, results: Dict[str, Dict], metric: str = 'accuracy'):
        """Построение графика сравнения"""
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        values = [results[m][metric] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values)
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt
    
    def generate_report(self, results: Dict[str, Dict], output_path: str = 'reports'):
        """Генерация отчета в PDF"""
        import pandas as pd
        from fpdf import FPDF
        import os
        
        os.makedirs(output_path, exist_ok=True)
        
        # Создание PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Model Comparison Report", ln=True, align='C')
        pdf.ln(10)
        
        # Таблица сравнения
        df = self.generate_comparison_table(results)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Model", 1)
        pdf.cell(30, 10, "Accuracy", 1)
        pdf.cell(30, 10, "Precision", 1)
        pdf.cell(30, 10, "Recall", 1)
        pdf.cell(30, 10, "F1 Score", 1)
        pdf.cell(35, 10, "Inference (ms)", 1)
        pdf.ln()
        
        pdf.set_font("Arial", "", 10)
        for _, row in df.iterrows():
            pdf.cell(40, 10, str(row['Model']), 1)
            pdf.cell(30, 10, f"{row['Accuracy']:.4f}", 1)
            pdf.cell(30, 10, f"{row['Precision']:.4f}", 1)
            pdf.cell(30, 10, f"{row['Recall']:.4f}", 1)
            pdf.cell(30, 10, f"{row['F1 Score']:.4f}", 1)
            pdf.cell(35, 10, f"{row['Inference (ms)']:.2f}", 1)
            pdf.ln()
        
        # Сохранение
        pdf_file = os.path.join(output_path, 'model_comparison_report.pdf')
        pdf.output(pdf_file)
        print(f"Report saved to {pdf_file}")
        
        # Сохранение в Excel
        excel_file = os.path.join(output_path, 'model_comparison.xlsx')
        df.to_excel(excel_file, index=False)
        print(f"Excel saved to {excel_file}")
        
        # Сохранение JSON
        json_file = os.path.join(output_path, 'results.json')
        with open(json_file, 'w') as f:
            # Удаляем предсказания и метки из JSON
            clean_results = {}
            for name, data in results.items():
                clean_results[name] = {k: v for k, v in data.items() 
                                     if k not in ['predictions', 'labels']}
            json.dump(clean_results, f, indent=2)
        print(f"JSON saved to {json_file}")