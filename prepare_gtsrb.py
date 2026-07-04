# prepare_gtsrb.py
import os
import shutil
from pathlib import Path

def prepare_gtsrb_data():
    # Пути
    train_dir = "Train"
    test_dir = "Test"
    processed_dir = "data/processed/gtsrb"
    
    # Создаем папку для обработанных данных
    os.makedirs(processed_dir, exist_ok=True)
    
    # Копируем тренировочные данные
    print("Copying training data...")
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if os.path.isdir(class_path):
            # Создаем папку класса в processed
            dest_class_path = os.path.join(processed_dir, class_dir)
            os.makedirs(dest_class_path, exist_ok=True)
            
            # Копируем изображения
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.ppm', '.jpg', '.png')):
                    src = os.path.join(class_path, img_file)
                    dst = os.path.join(dest_class_path, img_file)
                    shutil.copy2(src, dst)
            
            print(f"  Class {class_dir}: {len(os.listdir(dest_class_path))} images")
    
    # Копируем тестовые данные
    print("\nCopying test data...")
    if os.path.exists(test_dir):
        test_subdir = os.path.join(test_dir, "GT-final_test", "Images")
        if os.path.exists(test_subdir):
            # Создаем папки для тестовых классов
            for class_id in range(43):
                class_str = f"{class_id:05d}"
                dest_class_path = os.path.join(processed_dir, class_str)
                os.makedirs(dest_class_path, exist_ok=True)
            
            # Копируем тестовые изображения
            for img_file in os.listdir(test_subdir):
                if img_file.endswith('.ppm'):
                    # Имя файла содержит номер класса в начале
                    class_id = int(img_file.split('_')[0])
                    class_str = f"{class_id:05d}"
                    src = os.path.join(test_subdir, img_file)
                    dst = os.path.join(processed_dir, class_str, img_file)
                    shutil.copy2(src, dst)
            
            print(f"  Test images copied to respective class folders")
    
    print(f"\n✅ Data prepared in {processed_dir}")
    
    # Статистика
    total_images = 0
    for class_dir in os.listdir(processed_dir):
        class_path = os.path.join(processed_dir, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.ppm', '.jpg', '.png'))])
            total_images += count
            print(f"  Class {class_dir}: {count} images")
    
    print(f"\nTotal images: {total_images}")

if __name__ == "__main__":
    prepare_gtsrb_data()
