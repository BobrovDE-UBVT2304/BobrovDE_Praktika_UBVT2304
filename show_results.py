# show_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def show_results():
    # Загрузка результатов
    if not os.path.exists('reports/results.json'):
        print("❌ Результаты не найдены! Сначала запустите обучение.")
        return
    
    with open('reports/results.json', 'r') as f:
        results = json.load(f)
    
    # Создаем DataFrame
    df = pd.DataFrame(results).T
    df = df.reset_index().rename(columns={'index': 'Model'})
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("="*60)
    print(df[['Model', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    # Находим лучшую модель
    best_model = df.loc[df['accuracy'].idxmax()]
    print("\n" + "="*60)
    print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model['Model']}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   F1 Score: {best_model['f1_score']:.4f}")
    print("="*60)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График сравнения моделей
    ax1 = axes[0]
    models = df['Model'].values
    accuracies = df['accuracy'].values
    
    bars = ax1.bar(models, accuracies, color=['#4CAF50' if m == best_model['Model'] else '#2196F3' for m in models])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Сравнение моделей')
    ax1.set_ylim([0, 1.1])
    ax1.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # График размера модели и времени
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.35
    
    sizes = df['model_size_mb'].values / 10  # Нормализация для отображения
    times = df['avg_inference_time_ms'].values / 100  # Нормализация для отображения
    
    ax2.bar(x - width/2, sizes, width, label='Размер (M)', color='#FF9800')
    ax2.bar(x + width/2, times, width, label='Время (ms/100)', color='#9C27B0')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_title('Размер и скорость моделей')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('reports/results_visualization.png', dpi=150)
    plt.show()
    
    print("\n✅ Визуализация сохранена в reports/results_visualization.png")

if __name__ == "__main__":
    show_results()
