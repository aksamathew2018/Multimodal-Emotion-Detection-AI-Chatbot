# 🤖 Emotion AI Chatbot

A multimodal chatbot that detects user emotions from text (and optionally images) and responds in a natural, empathetic way.

---

## 🚀 Features

* 🧠 Emotion detection from text using DistilBERT
* 💬 Context-aware and emotion-aware chatbot responses
* 🖼 Optional image emotion detection (ResNet18)
* ⚡ Fast and lightweight (no heavy APIs required)
* 🌐 Deployable using Streamlit or Gradio

---

## 🛠 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* OpenCV
* Streamlit / Gradio

---

## 📊 Models Used

* **DistilBERT** → Multi-label text emotion classification (28 emotions)
* **ResNet18** → Image-based emotion detection (optional)
* **Rule-based logic** → Response generation

---

## 📂 Project Structure

```
Emotion-AI-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── Multimodal_emotion_detecting_bot.ipynb
├── text_model/        # trained model (optional)
├── image_model/       # image model (optional)
└── labels/
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Example

```
You: I feel very lonely these days  
Emotion: sad  
Bot: I'm really sorry you're feeling this way. Do you want to talk about it?
```

---

## ⚠️ Notes

* Image model is optional
* Works fully with text-based emotion detection
* Optimized for performance and simplicity
* No paid APIs required

---

## 🔮 Future Improvements

* 🎤 Voice input
* 🧠 Memory (context awareness)
* 🌙 Improved UI
* 🤖 Advanced LLM integration

---

## 👩‍💻 Author

**Aksa Mathew**

---

⭐ If you like this project, give it a star!
