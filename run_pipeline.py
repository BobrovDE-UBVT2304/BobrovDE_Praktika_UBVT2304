# run_pipeline.py
import os
import sys
from datetime import datetime

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_training():
    print("\n" + "="*60)
    print("STARTING TRAINING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    from main import train_all_models
    trained_models, all_results, best_model_name = train_all_models()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return best_model_name

def main():
    # Создаем необходимые папки
    os.makedirs('data/processed/gtsrb', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    data_path = 'data/processed/gtsrb'
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print(f"\n⚠️ No data found in {data_path}")
        print("Using dummy data for testing.")
        print("\nTo use real data:")
        print("  data/processed/gtsrb/class_id/image.jpg")
    
    best_model = run_training()
    
    if best_model:
        print(f"\n✅ Best model: {best_model}")
    else:
        print("\n⚠️ No models trained")

if __name__ == "__main__":
    main()