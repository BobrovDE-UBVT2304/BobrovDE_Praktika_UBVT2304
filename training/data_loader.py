# training/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import os
from PIL import Image
import random

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_per_class=50):  # Ограничиваем 50 изображений на класс
        self.data_dir = data_dir
        self.transform = transform
        self.max_per_class = max_per_class
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        if not os.path.exists(self.data_dir):
            print(f"Warning: {self.data_dir} does not exist")
            return
        
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Found {len(class_dirs)} class directories")
        
        for class_dir in class_dirs:
            class_path = os.path.join(self.data_dir, class_dir)
            try:
                class_id = int(class_dir)
            except ValueError:
                continue
            
            # Собираем все изображения в классе
            class_images = []
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg')):
                    class_images.append(os.path.join(class_path, img_name))
            
            # Ограничиваем количество изображений на класс
            if len(class_images) > self.max_per_class:
                class_images = random.sample(class_images, self.max_per_class)
            
            for img_path in class_images:
                self.images.append(img_path)
                self.labels.append(class_id)
        
        print(f"Loaded {len(self.images)} images (max {self.max_per_class} per class)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            image = Image.new('RGB', (128, 128), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(data_dir, batch_size=64, img_size=(128, 128)):
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        return create_dummy_loaders(batch_size)
    
    # Проверяем, есть ли данные
    has_data = False
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)):
            try:
                int(d)
                has_data = True
                break
            except:
                continue
    
    if not has_data:
        print("No data found. Creating dummy data...")
        return create_dummy_loaders(batch_size)
    
    # Трансформации с аугментацией для обучения
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=5, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Создаем полный датасет (ограничиваем 50 изображений на класс)
    full_dataset = TrafficSignDataset(data_dir, transform=train_transform, max_per_class=50)
    
    if len(full_dataset) == 0:
        print("No images found in dataset!")
        return create_dummy_loaders(batch_size)
    
    # Разбиение на train/val/test
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Применяем разные трансформации
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} images")
    print(f"  Val: {val_size} images")
    print(f"  Test: {test_size} images")
    print(f"  Total: {total} images")
    
    return train_loader, val_loader, test_loader

def create_dummy_loaders(batch_size=64):
    dummy_data = torch.randn(100, 3, 128, 128)
    dummy_labels = torch.randint(0, 43, (100,))
    
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Dummy data: Train={train_size}, Val={val_size}, Test={test_size}")
    return train_loader, val_loader, test_loader
