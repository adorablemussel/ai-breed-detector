import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import base64
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from model import get_model

CLASSES_JSON = "classes.json"
MODEL_PATH = "best_model.pth"
NELA_IMG_PATH = "Nela.png"
PAW_IMG_PATH = "PawBG.png"

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
        html_part = f"""
        <div class="nela-container">
                <div class="speech-bubble">{speech_text}</div>
                <img src="{NELA_B64}" class="nela-avatar" alt="Weterynarz Nela">
        </div>
        """

        return html_part

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
        return generate_nela_html(speech), [], gr.update(interactive=False)
        
    image = image.convert("RGB")
    tensor = predict_transforms(image).unsqueeze(0).to(device)
    
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
        
    return generate_nela_html(speech), results, gr.update(interactive=True)

def show_details(saved_results):
    if not saved_results:
        return generate_nela_html("Ups, brakuje mi wyników badań! Wgraj zdjęcie ponownie.")
        
    speech = "Jasne, oto dokładne wyniki z laboratorium:<br><br>"
    for breed, conf in saved_results:
        speech += f"🐾 <b>{breed}</b>: {conf * 100:.1f}%<br>"
        
    return generate_nela_html(speech)

def on_image_change(image):
    if image is None:
        speech = "Cześć! Jestem Nela. Wgraj zdjęcie pieska, a przekonamy się, jakiej jest rasy!"
    else:
        speech = "O, widzę nowego pacjenta! Kliknij <b>Rozpoznaj rasę</b>, abym mogła mu się przyjrzeć."
        
    return generate_nela_html(speech), [], gr.update(interactive=False)

NELA_JS = """
(function(){
    if (window.__nelaBubbleInstalled) return;
    window.__nelaBubbleInstalled = true;
    function updateBubble(){
        const nela = document.querySelector('.nela-container');
        if(!nela) return;
        const bubble = nela.querySelector('.speech-bubble');
        if(!bubble) return;
        const shouldFloatBelow = window.innerWidth <= 768 && window.scrollY > 24;
        if(shouldFloatBelow){
            nela.classList.add('bubble-below');
            bubble.classList.add('bubble-below');
        } else {
            nela.classList.remove('bubble-below');
            bubble.classList.remove('bubble-below');
        }
    }
    window.addEventListener('scroll', updateBubble, {passive:true});
    window.addEventListener('resize', updateBubble);
    const mo = new MutationObserver(()=> setTimeout(updateBubble, 100));
    mo.observe(document.body, {childList:true, subtree:true, attributes:true});
    setInterval(updateBubble, 100);
    setTimeout(updateBubble, 400);
})();
"""

NELA_HEAD = f"<script>{NELA_JS}</script>"

customtheme = gr.themes.Soft(primary_hue="sky", neutral_hue='slate').set(
    background_fill_primary="#0B132B",
    background_fill_secondary="#1C2541",
    block_background_fill="#1C2541",
    body_text_color="#E2E8F0",
    block_label_text_color="#E2E8F0",
    block_label_background_fill="#243388",
    block_label_border_color="#243388", 
)

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
    
    input_img.change(
        fn=on_image_change,
        inputs=input_img,
        outputs=[nela_display, current_results, btn_details]
    )

    btn_predict.click(
        fn=predict_dog_breed, 
        inputs=input_img, 
        outputs=[nela_display, current_results, btn_details]
    )
    
    btn_details.click(
        fn=show_details,
        inputs=current_results,
        outputs=nela_display
    )
    

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=customtheme, css=customcss, head=NELA_HEAD)