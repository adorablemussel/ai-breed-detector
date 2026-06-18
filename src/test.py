import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

from model import get_model

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / "data" / "processed" / "test"
CLASSES_JSON = BASE_DIR / "data" / "processed" / "classes.json"
MODEL_PATH = BASE_DIR / "saved_models" / "best_model.pth"

REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

def evaluate_model():
    print("Rozpoczynam ewaluację modelu na zbiorze testowym...")

    # 1. Wybór urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie obliczeniowe: {device}")

    # 2. Ładowanie mapowania klas
    with open(CLASSES_JSON, "r") as f:
        breed_to_id = json.load(f)
    id_to_breed = {v: k for k, v in breed_to_id.items()}
    class_count = len(breed_to_id)

    # 3. Transformacje dla danych testowych
    test_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Ładowanie danych testowych
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Budowa modelu z pliku i załadowanie wag z dysku
    model = get_model(class_count=class_count, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() # Tryb oceny - BARDZO WAŻNE

    # Listy do przechowywania wyników
    all_preds = []
    all_labels = []
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    print("Obliczanie predykcji...")
    
    # 6. PĘTLA EWALUACYJNA
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            # --- Top-1 Accuracy ---
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            # --- Top-3 Accuracy ---
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top3_preds[i]:
                    correct_top3 += 1

            total += labels.size(0)

            # Zbieranie danych do macierzy pomyłek
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. OBLICZANIE METRYK
    top1_acc = 100 * correct_top1 / total
    top3_acc = 100 * correct_top3 / total

    print(f"\n--- WYNIKI NA ZBIORZE TESTOWYM ---")
    print(f"Skutecznosc (Top-1 Accuracy): {top1_acc:.2f}%")
    print(f"Skutecznosc Top-3: {top3_acc:.2f}%")

    # Zapis raportu do pliku
    report_path = REPORTS_DIR / "test_metrics_report.txt"
    with open(report_path, "w") as f:
        f.write("RAPORT Z EWALUACJI MODELU\n")
        f.write("="*30 + "\n")
        f.write(f"Skutecznosc ogolna (Top-1): {top1_acc:.2f}%\n")
        f.write(f"Skutecznosc Top-3: {top3_acc:.2f}%\n")
        f.write(f"Przetestowano na lacznie: {total} obrazach.\n\n")
        
        f.write(classification_report(all_labels, all_preds, target_names=[id_to_breed[i] for i in range(class_count)]))
    
    print(f"Raport zapisano do: {report_path}")

    # 8. MACIERZ POMYŁEK
    print("\nGenerowanie Macierzy Pomyłek...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)
    plt.title('Macierz Pomyłek (Confusion Matrix)', fontsize=40)
    plt.xlabel('Przewidziana Klasa', fontsize=30)
    plt.ylabel('Prawdziwa Klasa', fontsize=30)
    
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=100)
    print(f"Macierz pomyłek zapisano do: {cm_path}")

if __name__ == "__main__":
    evaluate_model()