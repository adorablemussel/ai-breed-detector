import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from model import get_model

# UPROSZCZONA KONFIGURACJA ŚCIEŻEK DLA HUGGING FACE
CLASSES_JSON = "classes.json"
MODEL_PATH = "best_model.pth"

# ładowanie nazw ras (Słownik id -> nazwa)
try:
    with open(CLASSES_JSON, "r") as f:
        breed_to_id = json.load(f)
    id_to_breed = {v: k.replace('_', ' ').title() for k, v in breed_to_id.items()}
    class_count = len(id_to_breed)
except Exception as e:
    print(f"Błąd ładowania pliku classes.json: {e}")
    class_count = 120

# inicjalizacja modelu
# na darmowym koncie HF mamy tylko CPU
device = torch.device("cpu")
print(f"Uruchamiam aplikację na urządzeniu: {device}")

# ładowanie architektury
model = get_model(class_count=class_count, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# transformacja zdjęcia
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# funkcja do klasyfikacji dla gradio
def predict_dog_breed(image):
    if image is None:
        return None
        
    # <-- ZABEZPIECZENIE: Konwersja obrazu do 3 kanałów (RGB) 
    # chroni przed crashem przy zdjęciach czarno-białych lub z przezroczystym tłem (PNG)
    image = image.convert("RGB")
    
    tensor = predict_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0) # obliczanie prawdopodobieństwa

    top3_prob, top3_indices = torch.topk(probabilities, 3)
    # słownik wyników dla Gradio: { "nazwa_klasy": wartość }
    results = {}
    for i in range(3):
        index = top3_indices[i].item()
        
        # Zabezpieczenie na wypadek, gdyby klucze w JSON odczytały się jako stringi
        if index in id_to_breed:
            breed_name = id_to_breed[index]
        else:
            breed_name = id_to_breed[str(index)]
            
        confidence = top3_prob[i].item()
        results[breed_name] = confidence
        
    return results

# budowa interfejsu gradio wraz ze stylizacją
customtheme = gr.themes.Soft(primary_hue="sky", neutral_hue='slate').set(
    background_fill_primary="#0B132B",
    background_fill_secondary="#1C2541",
    block_background_fill="#1C2541",
    body_text_color="#E2E8F0",
    block_label_text_color="#E2E8F0",
    block_label_background_fill="#243388",
    block_label_border_color="#243388", 
)

customcss = """
.title {
    text-align: center;
    color: #E2E8F0;
    font-size: 2.8rem;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
"""

with gr.Blocks() as demo:
    gr.HTML("<h1 class='title'>AI Dog Breed Detector</h1>")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Wgraj zdjęcie psa")
            btn_predict = gr.Button("Rozpoznaj rasę", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(num_top_classes=3, label="Najbardziej prawdopodobna rasa psa")
    
    btn_predict.click(fn=predict_dog_breed, inputs=input_img, outputs=output_labels)

if __name__ == "__main__":
    demo.launch(share=False, theme=customtheme, css=customcss)