import os
import shutil

# Ana klasör yolunu belirtin
SOURCE_DIR = "/Users/zehraatalay/Desktop/dataset/validation"  # Mevcut resimlerin bulunduğu klasör
DEST_DIR = "/Users/zehraatalay/Desktop/dataset/validation"  # Ayrıştırılmış resimlerin taşınacağı ana klasör

# Klasör oluşturma fonksiyonu
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ana işlem
def organize_images_by_label(source_dir, dest_dir):
    # Kaynak klasördeki dosyaları al
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Her dosya için işlemleri gerçekleştir
    for file in files:
        # Dosyanın ilk harfini al
        label = file[0].upper()  # İlk harfi büyük harfe çevir

        # Hedef klasör yolunu oluştur
        target_folder = os.path.join(dest_dir, label)
        create_dir_if_not_exists(target_folder)  # Eğer klasör yoksa oluştur

        # Dosyayı hedef klasöre taşı
        shutil.move(os.path.join(source_dir, file), os.path.join(target_folder, file))

    print("Dosyalar isimlerinin ilk harfine göre organize edildi!")

# Fonksiyonu çalıştır
organize_images_by_label(SOURCE_DIR, DEST_DIR)
