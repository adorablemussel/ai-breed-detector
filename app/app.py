import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import base64
import numpy as np
import cv2
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO


CLASSES_JSON = "classes.json"
MODEL_PATH = "best_model.pth"
NELA_IMG_PATH = "Nela.png"
PAW_IMG_PATH = "PawBG.png"

yolo_model = YOLO('yolo11n.pt') 

def is_dog_present(image):
    results = yolo_model(image)
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 16:
                return True
    return False

try:
    with open(PAW_IMG_PATH, "rb") as f:
        png_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><image href="{png_b64}" x="60" y="60" width="80" height="80"/></svg>"""
        PAW_B64 = "data:image/svg+xml;base64," + base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
except Exception as e:
    print(f"Nie znaleziono zdjęcia tła: {e}")
    PAW_B64 = ""

try:
    with open(NELA_IMG_PATH, "rb") as f:
        NELA_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')
except Exception as e:
    print(f"Nie znaleziono zdjęcia awatara: {e}")
    NELA_B64 = ""

def generate_nela_html(speech_text):
    return f"""
    <div class="nela-container">
            <div class="speech-bubble">{speech_text}</div>
            <img src="{NELA_B64}" class="nela-avatar" alt="Weterynarz Nela">
    </div>
    """

try:
    with open(CLASSES_JSON, "r") as f:
        breed_to_id = json.load(f)
    id_to_breed = {v: k.replace('_', ' ').title() for k, v in breed_to_id.items()}
    class_count = len(id_to_breed)
except Exception as e:
    print(f"Błąd ładowania pliku classes.json: {e}")
    class_count = 120

device = torch.device("cpu")
print(f"Uruchamiam aplikację na urządzeniu: {device}")

model = get_model(class_count=class_count, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_dog_breed(image):
    if image is None:
        speech = "Proszę, wgraj najpierw zdjęcie pieska! Nie potrafię diagnozować powietrza."
        return generate_nela_html(speech), None, gr.update(interactive=False), gr.update(label="Wgraj zdjęcie psa"), gr.update(interactive=True)

    if not is_dog_present(image):
        speech = "Ojej, obawiam się, że na tym zdjęciu nie widzę żadnego pieska! Nie przeceniaj mnie, jestem tylko weterynarzem."
        return generate_nela_html(speech), None, gr.update(interactive=False), gr.update(label="Wgraj zdjęcie psa"), gr.update(interactive=True)

    image_rgb = image.convert("RGB")
    tensor = predict_transforms(image_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        index = top3_indices[i].item()
        if index in id_to_breed:
            breed_name = id_to_breed[index]
        else:
            breed_name = id_to_breed[str(index)]
        confidence = top3_prob[i].item()
        results.append((breed_name, confidence))
        
    top_breed, top_conf = results[0]
    
    if top_conf > 0.75:
        speech = f"Gotowe! Z moich weterynaryjnych oględzin wynika, że to <b>rasowy piesek</b>!<br><br>To na pewno <b>{top_breed}</b>."
    else:
        breed2 = results[1][0]
        breed3 = results[2][0]
        speech = f"Gotowe! Wygląda na to, że mamy tu uroczego <b>kundelka</b>!<br><br>Moje bystre oko widzi tu mieszankę genów. Najwięcej cech odziedziczył po: <br><b>{top_breed}</b>, <b>{breed2}</b> oraz <b>{breed3}</b>."
        
    state_data = {
        'results': results,
        'image': image_rgb,
        'tensor': tensor,
        'top_index': top3_indices[0].item()
    }
        
    return generate_nela_html(speech), state_data, gr.update(interactive=True), gr.update(label="Wgraj zdjęcie psa"), gr.update(interactive=True)

def show_details(saved_state):
    if not saved_state:
        return generate_nela_html("Ups, brakuje mi wyników badań! Wgraj zdjęcie ponownie."), gr.update(), gr.update()
        
    results = saved_state['results']
    img_pil = saved_state['image']
    tensor = saved_state['tensor']
    top_index = saved_state['top_index']
    
    speech = "Jasne, oto dokładne wyniki z laboratorium oraz mapa cieplna pokazująca na co zwróciłam uwagę:<br><br>"
    for breed, conf in results:
        speech += f"🐾 <b>{breed}</b>: {conf * 100:.1f}%<br>"
        
    try:
        target_layers = [model.features[-1][-1]]
        
        with GradCAM(model=model, target_layers=target_layers) as cam:
            targets = [ClassifierOutputTarget(top_index)]
            # Otrzymujemy maskę kwadratową (224x224)
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
            
        orig_width, orig_height = img_pil.size
        
        grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_width, orig_height))
        
        img_np = np.array(img_pil) / 255.0
        
        heatmap_visualization = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)
        
        return generate_nela_html(speech), gr.update(value=heatmap_visualization, label="Mapa cieplna (Grad-CAM)"), gr.update(interactive=False)
        
    except Exception as e:
        print(f"Błąd przy generowaniu XAI (Grad-CAM): {e}")
        speech += "<br><br><i>Niestety wystąpił problem z wygenerowaniem mapy cieplnej.</i>"
        return generate_nela_html(speech), gr.update(), gr.update()

def on_image_upload():
    speech = "O, widzę nowego pacjenta! Kliknij <b>Rozpoznaj rasę</b>, abym mogła mu się przyjrzeć."
    return generate_nela_html(speech), None, gr.update(interactive=False), gr.update(label="Wgraj zdjęcie psa"), gr.update(interactive=True)

def on_image_clear():
    speech = "Cześć! Jestem Nela. Wgraj zdjęcie pieska, a przekonamy się, jakiej jest rasy!"
    return generate_nela_html(speech), None, gr.update(interactive=False), gr.update(label="Wgraj zdjęcie psa"), gr.update(interactive=True)

NELA_JS = """
(function(){
    if (window.__nelaBubbleInstalled) return;
    window.__nelaBubbleInstalled = true;
    function updateBubble(){
        const nela = document.querySelector('.nela-container');
        if(!nela) return;
        const bubble = nela.querySelector('.speech-bubble');
        if(!bubble) return;
        
        let scrollY = window.scrollY || document.documentElement.scrollTop || 0;
        const gradioWrap = document.querySelector('.gradio-container');
        if(gradioWrap) scrollY = Math.max(scrollY, gradioWrap.scrollTop);
        const rect = nela.getBoundingClientRect();
        
        const shouldFloatBelow = window.innerWidth <= 1700 && (scrollY > 24 || rect.top < 60);
        if(shouldFloatBelow){
            nela.classList.add('bubble-below');
            bubble.classList.add('bubble-below');
        } else {
            nela.classList.remove('bubble-below');
            bubble.classList.remove('bubble-below');
        }
    }
    window.addEventListener('scroll', updateBubble, true);
    window.addEventListener('resize', updateBubble);
    setInterval(updateBubble, 100);
})();
"""

customtheme = gr.themes.Soft(primary_hue="sky", neutral_hue='slate').set(
    background_fill_primary="#0B132B",
    background_fill_secondary="#1C2541",
    block_background_fill="#1C2541",
    body_text_color="#E2E8F0",
    block_label_text_color="#E2E8F0",
    block_label_background_fill="#243388",
    block_label_border_color="#243388", 
)

css_file_path = Path(__file__).parent / "style.css"
try:
    with open(css_file_path, "r", encoding="utf-8") as f:
        customcss = f.read()
        customcss = customcss.replace("{{PAW_B64}}", PAW_B64)
except Exception as e:
    print(f"Błąd ładowania pliku CSS: {e}")
    customcss = ""

with gr.Blocks() as demo:
    gr.HTML("<h1 class='title'>Poznaj Rasę Psa</h1>")

    current_results = gr.State()

    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(scale=1, elem_classes=["upload-column"]):
            input_img = gr.Image(
                type="pil",
                label="Wgraj zdjęcie psa",
                height=540,
                elem_classes=["upload-image"],
            )
            
            with gr.Row(elem_classes=["upload-actions"]):
                btn_predict = gr.Button("Rozpoznaj rasę", variant="primary")
                btn_details = gr.Button("Zapytaj o szczegóły badań", variant="secondary", interactive=False)
        
        with gr.Column(scale=1, elem_classes=["nela-column"]):
            powitanie = "Cześć! Jestem Nela. Wgraj zdjęcie pieska, a przekonamy się, jakiej jest rasy!"
            nela_display = gr.HTML(generate_nela_html(powitanie))
    
    input_img.upload(
        fn=on_image_upload,
        inputs=None,
        outputs=[nela_display, current_results, btn_details, input_img, btn_predict]
    )
    
    input_img.clear(
        fn=on_image_clear,
        inputs=None,
        outputs=[nela_display, current_results, btn_details, input_img, btn_predict]
    )

    btn_predict.click(
        fn=predict_dog_breed, 
        inputs=input_img, 
        outputs=[nela_display, current_results, btn_details, input_img, btn_predict]
    )
    
    btn_details.click(
        fn=show_details,
        inputs=current_results,
        outputs=[nela_display, input_img, btn_predict]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=customtheme, css=customcss, js=NELA_JS)