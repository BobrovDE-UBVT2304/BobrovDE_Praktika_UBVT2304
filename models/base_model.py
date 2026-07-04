# models/base_model.py

class BaseModel:
    def __init__(self, num_classes, input_size=(224, 224)):
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None
        self.name = "BaseModel"
    
    def get_model(self):
        if self.model is None:
            self.model = self.build_model()
        return self.model
    
    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def get_preprocessing(self):
        raise NotImplementedError("Subclasses must implement get_preprocessing()")
