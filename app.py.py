import gradio as gr
import random
import torch
import cv2
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

BASE_PATH = "./"

with open(f"{BASE_PATH}/labels/text_labels.json") as f:
    text_labels = json.load(f)

with open(f"{BASE_PATH}/labels/image_labels.json") as f:
    image_labels = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/text_model")
text_model = AutoModelForSequenceClassification.from_pretrained(f"{BASE_PATH}/text_model")
text_model.eval()

image_model = models.resnet18()
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(image_labels))
image_model.load_state_dict(torch.load(f"{BASE_PATH}/image_model/resnet_model.pth", map_location="cpu"))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def map_to_fer(label):
    if label in ['anger', 'annoyance']:
        return 'angry'
    elif label in ['joy', 'amusement', 'love']:
        return 'happy'
    elif label in ['sadness', 'grief']:
        return 'sad'
    else:
        return 'neutral'

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return map_to_fer(text_labels[idx])

def predict_image(image_path):
    image_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image_cv[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face)
    else:
        image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    idx = torch.argmax(probs).item()
    return image_labels[idx]

def adjust_emotion(user_input, predicted):
    text = user_input.lower()

    if "exhaust" in text or "tired" in text:
        return "sad"
    if "angry" in text:
        return "angry"
    if "job" in text or "happy" in text:
        return "happy"

    return predicted

def generate_response(user_input, emotion):
    if "thank" in user_input.lower():
        return "You're welcome 😊"

    if emotion == "happy":
        return "That's amazing! Tell me more!"
    elif emotion == "sad":
        return "I'm sorry you're feeling this way. Want to talk about it?"
    elif emotion == "angry":
        return "That sounds frustrating. What happened?"
    else:
        return "I see. Tell me more."

def chat(message, image, history):
    if image:
        emotion = predict_image(image)
        response = f"You look {emotion}. Tell me more."
        user = "📷 Image"
    else:
        pred = predict_text(message)
        emotion = adjust_emotion(message, pred)
        response = generate_response(message, emotion)
        user = message

    history.append((user, f"{emotion} → {response}"))
    return "", None, history

with gr.Blocks() as demo:
    gr.Markdown("# Emotion AI Chatbot")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    img = gr.Image(type="filepath")

    btn = gr.Button("Send")

    btn.click(chat, [msg, img, chatbot], [msg, img, chatbot])

demo.launch()
