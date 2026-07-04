# main.py
import sys
import os
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.config import get_gtsrb_config
from training.trainer import Trainer
from training.data_loader import create_data_loaders
from training.metrics import ModelEvaluator
from models.model_factory import ModelFactory

def train_all_models():
    config = get_gtsrb_config()
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data_path, config.batch_size, config.img_size
    )
    
    # Выбираем только 3-4 модели для быстрого обучения
    # model_names = ModelFactory.get_available_models()  # Все 7 моделей
    model_names = ['simple_cnn', 'residual_cnn', 'efficientnet_b0', 'mobilenet_v3']  # Только 4 модели
    
    print(f"\nTraining {len(model_names)} models: {model_names}")
    
    trained_models = {}
    all_results = {}
    best_model_name = None
    best_accuracy = 0.0
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print('='*60)
        
        try:
            model_interface = ModelFactory.create_model(model_name, config.num_classes, config.img_size)
            model = model_interface.get_model()
            
            trainer = Trainer(model, config)
            
            trainer.train(train_loader, val_loader, epochs=config.num_epochs,
                         lr=config.learning_rate, weight_decay=config.weight_decay)
            
            model_path = f"{config.save_dir}/{model_name}_final.pth"
            trainer.save_model(model_path)
            
            test_results = trainer.test(test_loader)
            test_results['model_name'] = model_name
            test_results['model_path'] = model_path
            
            all_results[model_name] = test_results
            trained_models[model_name] = model
            
            if test_results['accuracy'] > best_accuracy:
                best_accuracy = test_results['accuracy']
                best_model_name = model_name
            
            print(f"✅ {model_name} completed! Accuracy: {test_results['accuracy']:.4f}")
            
            # Очищаем память
            del model
            del trainer
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if trained_models:
        evaluator = ModelEvaluator()
        comparison_results = evaluator.compare_models(trained_models, test_loader)
        comparison_df = evaluator.generate_comparison_table(comparison_results)
        
        print("\n" + "="*60)
        print("COMPARISON TABLE")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        with open('reports/all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        evaluator.generate_report(comparison_results, 'reports')
        
        print(f"\n✅ Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return trained_models, all_results, best_model_name

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAFFIC SIGN RECOGNITION - TRAINING")
    print("="*60)
    print("Optimized for speed: 4 models, 5 epochs, 128x128 images")
    print("="*60 + "\n")
    
    trained_models, all_results, best_model_name = train_all_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    if all_results:
        print(f"Best model: {best_model_name}")
        print(f"Best accuracy: {max([r['accuracy'] for r in all_results.values()]):.4f}")
    print("="*60)
