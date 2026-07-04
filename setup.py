# setup.py
from setuptools import setup, find_packages

setup(
    name="traffic_sign_recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.5',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'opencv-python>=4.5.0',
        'pillow>=8.0.0',
        'tqdm>=4.62.0',
        'tensorboard>=2.6.0',
        'openpyxl>=3.0.0',
        'python-dotenv>=0.19.0',
        'fpdf>=1.7.2',
        'streamlit>=1.0.0',
        'plotly>=5.0.0'
    ],
)