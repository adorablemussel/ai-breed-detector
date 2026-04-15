import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import json

# Konfiguracja ścieżek
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images" / "Images"
ANNOT_DIR = RAW_DIR / "annotations" / "Annotation"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Proporcje podziału
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

def parse_xml(xml_path):
    """Wyciąga współrzędne bounding boxa z pliku XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Bierzemy pierwszy znaleziony obiekt psa
        obj = root.find("object")
        bndbox = obj.find("bndbox")
        
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        return (xmin, ymin, xmax, ymax)
    except Exception as e:
        print(f"Błąd podczas parsowania {xml_path}: {e}")
        return None

def prepare_data():
    # Pobranie listy folderów (ras)
    breed_folders = sorted([f for f in os.listdir(IMAGES_DIR) if os.path.isdir(IMAGES_DIR / f)])
    
    all_data = []
    class_names = {}

    print(f"Znaleziono {len(breed_folders)} ras, początek indeksowania...")

    for folder in breed_folders:
        # Oczyszczanie nazwy ras
        clean_name = folder.split('-', 1)[1] if '-' in folder else folder
        class_names[folder] = clean_name
        
        breed_img_path = IMAGES_DIR / folder
        breed_ann_path = ANNOT_DIR / folder
        
        for img_name in os.listdir(breed_img_path):
            img_id = os.path.splitext(img_name)[0]
            
            # Ścieżki do plików
            img_path = breed_img_path / img_name
            ann_path = breed_ann_path / img_id
            
            if img_path.exists() and ann_path.exists():
                all_data.append({
                    "img_path": img_path,
                    "ann_path": ann_path,
                    "breed": clean_name
                })

    # Podział danych Stratified Split (podział strategiczny dla każdej rasy)
    # Train 80%
    train_data, temp_data = train_test_split(
        all_data, 
        train_size=TRAIN_SIZE, 
        stratify=[d['breed'] for d in all_data],
        random_state=42
    )
    
    # Val i Test po 10% (pół na pół z pozostałych 20%)
    val_size_relative = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_data, test_data = train_test_split(
        temp_data, 
        train_size=val_size_relative, 
        stratify=[d['breed'] for d in temp_data],
        random_state=42
    )

    # Przetwarzanie i zapisywanie zbiorów
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split_name, data_list in datasets.items():
        print(f"Przetwarzanie zbioru: {split_name} ({len(data_list)} obrazów)...")
        
        for item in data_list:
            # Przygotowanie folderu docelowego
            target_dir = PROCESSED_DIR / split_name / item['breed']
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Parsowanie i wycinanie
            coords = parse_xml(item['ann_path'])
            if coords:
                try:
                    with Image.open(item['img_path']) as img:
                        # Wycięcie (crop) psa ze zdjęcia według współrzędnych z adnotacji
                        cropped_img = img.crop(coords)
                        # Konwersja do RGB
                        cropped_img = cropped_img.convert("RGB")
                        # Zapis
                        save_path = target_dir / item['img_path'].name
                        cropped_img.save(save_path, "JPEG")
                except Exception as e:
                    print(f"Błąd przy zapisie {item['img_path']}: {e}")

    # Zapisanie mapowania klas w pliku json
    with open(PROCESSED_DIR / "classes.json", "w") as f:
        # Sortowanie nazw ras i przypisanie id
        unique_breeds = sorted(list(set(class_names.values())))
        breed_to_id = {name: i for i, name in enumerate(unique_breeds)}
        json.dump(breed_to_id, f, indent=4)

    print("\nDane podzielone i zapisane w 'data/processed'.")
    print(f"Mapowanie klas zapisano w 'data/processed/classes.json'.")

if __name__ == "__main__":
    prepare_data()