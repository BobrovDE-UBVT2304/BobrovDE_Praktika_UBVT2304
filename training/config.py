# training/config.py

class TrainingConfig:
    def __init__(self):
        self.data_path = "data/processed/gtsrb"
        self.num_classes = 43
        self.img_size = (128, 128)  # Уменьшаем с 224 до 128
        self.batch_size = 64        # Увеличиваем batch size
        self.num_epochs = 5         # Уменьшаем до 5 эпох
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.use_augmentation = True
        self.save_dir = "checkpoints"
        self.log_dir = "logs"
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def get_gtsrb_config():
    return TrainingConfig()
