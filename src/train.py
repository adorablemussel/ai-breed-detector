import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_NAME = 'best_model.pth'
SAVE_PATH = BASE_DIR / "saved_models"

PLOT_NAME = 'training_results.png'
PLOT_DIR = BASE_DIR / "reports" / "plots"

BATCH_SIZE = 32         # - rozmiar jednej paczki do nauki
EPOCHS = 20             # - liczba przerobień całego datasetu
LEARNING_RATE = 0.001   # - krok spadku błędu przy optymalizacji

def train_model():
    # informacje o stałych
    print(f"Nazwa uczonego modelu: {MODEL_NAME}")
    print(f"Rozmiar pojedynczej partii: {BATCH_SIZE}")
    print(f"Liczba wszystkich epok: {EPOCHS}")
    print(f"Współczynnik uczenia się: {LEARNING_RATE}")

    # wybieramy jednostkę do przeprowadzenia uczenia
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Do treningu modelu wybrano: {device}")

    # dostosowanie zdjęć do nauki
    train_transforms_data = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # - zapobiega przeuczaniu się
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    train_transforms = transforms.Compose(train_transforms_data)

    val_transforms_data = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = transforms.Compose(val_transforms_data)

    # ładujemy dane
    train_dataset = datasets.ImageFolder(root=PROCESSED_DIR/"train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=PROCESSED_DIR/"val", transform=val_transforms)

    class_count = len(train_dataset.classes)
    print(f"Znaleziona liczba klas: {class_count}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # wybór modelu, nowy head na 120 wyjść (a nie defaultowe 1000), wysłanie do urządzenia
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # (można przetestować resnet50)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_count)
    model = model.to(device)

    # funkcja błędu i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # --- STRUKTURA DO PRZECHOWYWANIA STATYSTYK ---
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # PĘTLA UCZENIA MASZYNOWEGO
    print("\nROZPOCZYNAM UCZENIE")
    for i in range(EPOCHS):
        model.train() # - tryb nauki
        running_loss = float(0)
        correct_train = int(0)
        total_train = int(0)

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()               # 1 - zerowanie starych błędów
            outputs = model(images)             # 2 - predykcja
            loss = criterion(outputs, labels)   # 3 - obliczenie błędu
            loss.backward()                     # 4 - obliczenie kierunku poprawy
            optimizer.step()                    # 5 - aktualizacja wiedzy sieci

            # statystyki
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # train_acc = 100 * correct_train / total_train
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # walidacja (sprawdzenie modelu na danych, których nie widział)
        model.eval() # - tryb oceny
        val_loss = float(0)
        correct_val = int(0)
        total_val = int(0)

        with torch.no_grad(): # podczas testów nie aktualizujemy wag
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # val_acc = 100 * correct_val / total_val
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val

        # ZAPISYWANIE WYNIKÓW DO HISTORII 
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # print(f"Epoka {i+1}/{EPOCHS} | "
        #       f"Bład Treningu: {running_loss/len(train_loader):.4f} | Dokładność Treningu: {train_acc:.2f}% | "
        #       f"Bład Walidacji: {val_loss/len(val_loader):.4f} | Dokładność Walidacji: {val_acc:.2f}%")
        print(f"Epoka {i+1}/{EPOCHS} | "
              f"Loss T: {epoch_train_loss:.4f} | Acc T: {epoch_train_acc:.2f}% | "
              f"Loss V: {epoch_val_loss:.4f} | Acc V: {epoch_val_acc:.2f}%")
        
    # --- GENEROWANIE WYKRESÓW ---
    print("\nGenerowanie wykresów...")
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))

    # Wykres Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Błąd Treningu')
    plt.plot(epochs_range, history['val_loss'], label='Błąd Walidacji')
    plt.title('Funkcja Straty (Loss)')
    plt.xlabel('Epoki')
    plt.ylabel('Wartość błędu')
    plt.legend()

    # Wykres Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Dokładność Treningu')
    plt.plot(epochs_range, history['val_acc'], label='Dokładność Walidacji')
    plt.title('Dokładność Modelu (Accuracy)')
    plt.xlabel('Epoki')
    plt.ylabel('Procent poprawnych trafień')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / PLOT_NAME)
    print(f"Wykresy zostały zapisane jako: {PLOT_DIR / PLOT_NAME}")
        
    # zapis modelu
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH/MODEL_NAME)
    print(f"\nUkończono Proces Uczenia Maszynowego.\nModel został zapisany w: {SAVE_PATH/MODEL_NAME}")

if __name__ == "__main__":
    train_model()