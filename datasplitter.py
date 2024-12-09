import os
import shutil
import random

# Ana data klasörü
DATA_DIR = "/Users/zehraatalay/Desktop/data"

# Yeni klasörlerin yollarını belirleyin
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Eğer bu klasörler yoksa, oluştur
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Her alt klasörü işle
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Sadece alt klasörleri işle
    if os.path.isdir(dir_path) and not dir_.startswith('.') and dir_ not in ["train", "validation", "test"]:
        print(f"Processing folder: {dir_}")

        # Dosyaların tam yolunu al ve karıştır
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        random.shuffle(files)  # Dosyaları karıştır

        # Veriyi böl (80% train, 10% validation, 10% test)
        train_split = int(len(files) * 0.8)
        val_split = int(len(files) * 0.9)

        train_files = files[:train_split]
        validation_files = files[train_split:val_split]
        test_files = files[val_split:]

        # Hedef alt klasörlerini oluştur
        os.makedirs(os.path.join(TRAIN_DIR, dir_), exist_ok=True)
        os.makedirs(os.path.join(VALIDATION_DIR, dir_), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, dir_), exist_ok=True)

        # Train dosyalarını taşı
        for file in train_files:
            shutil.copy(os.path.join(dir_path, file), os.path.join(TRAIN_DIR, dir_, file))

        # Validation dosyalarını taşı
        for file in validation_files:
            shutil.copy(os.path.join(dir_path, file), os.path.join(VALIDATION_DIR, dir_, file))

        # Test dosyalarını taşı
        for file in test_files:
            shutil.copy(os.path.join(dir_path, file), os.path.join(TEST_DIR, dir_, file))

        print(f"Processed folder: {dir_}")
print("Data split into train, validation, and test folders successfully.")
