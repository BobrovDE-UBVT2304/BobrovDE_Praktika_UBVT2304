# clean_cache.py
import os
import shutil

def clean_pycache():
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                path = os.path.join(root, dir_name)
                print(f"Removing {path}")
                shutil.rmtree(path)
        for file in files:
            if file.endswith('.pyc'):
                path = os.path.join(root, file)
                print(f"Removing {path}")
                os.remove(path)

if __name__ == "__main__":
    clean_pycache()
    print("Cache cleaned!")