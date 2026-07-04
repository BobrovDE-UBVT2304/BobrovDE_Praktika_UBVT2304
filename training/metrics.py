# training/metrics.py
import torch
import pandas as pd
import json
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, test_loader, model_name):
        model.eval()
        device = next(model.parameters()).device
        all_preds, all_labels = [], []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                start = time.time()
                output = model(data)
                inference_times.append(time.time() - start)
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'model_size_mb': sum(p.numel() for p in model.parameters()) / 1e6,
            'avg_inference_time_ms': sum(inference_times) / len(inference_times) * 1000 if inference_times else 0
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models_dict, test_loader):
        results = {}
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, test_loader, name)
        return results
    
    def generate_comparison_table(self, results):
        df_data = []
        for model_name, metrics in results.items():
            df_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'Size (M)': metrics['model_size_mb'],
                'Inference (ms)': metrics['avg_inference_time_ms']
            })
        df = pd.DataFrame(df_data)
        return df.sort_values('Accuracy', ascending=False)
    
    def generate_report(self, results, output_path='reports'):
        os.makedirs(output_path, exist_ok=True)
        df = self.generate_comparison_table(results)
        df.to_csv(f'{output_path}/comparison.csv', index=False)
        with open(f'{output_path}/results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Report saved to {output_path}")
        return df
