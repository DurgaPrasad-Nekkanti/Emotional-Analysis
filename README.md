# 🧠 Emotion Detection using GRU (94% Accuracy)

This project performs **emotion classification** from text using a **GRU-based Recurrent Neural Network**. The model is trained on a public dataset containing user sentences labeled with emotions such as *joy, anger, sadness, etc.*, and achieves a test accuracy of **94%**.

---

## 📌 Problem Statement

Given a sentence as input, the task is to classify the underlying emotion into one of six categories:  
**[joy, sadness, anger, fear, love, surprise]**

---

## 📊 Dataset

- A publicly available dataset of 20,000+ labeled text samples
- Each text is annotated with one of six emotion labels
- Used for multiclass emotion classification

---

## 🧠 Model Architecture

The model is built using the **TensorFlow/Keras** deep learning framework:

- **Text Preprocessing**:
  - Tokenization
  - Padding sequences
- **Neural Network Layers**:
  - Embedding Layer
  - GRU (Gated Recurrent Unit)
  - Dropout Layer
  - Dense Output Layer with Softmax activation
- **Loss Function**: Sparse Categorical Crossentropy  
- **Optimizer**: Adam

---

## 📈 Evaluation Metrics

- ✅ Accuracy: 94% on validation data
- 📊 Visualizations:
  - Confusion Matrix
  - Accuracy & Loss Curves

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 🧪 How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DurgaPrasad-Nekkanti/Emotional-Analysis.git
   cd Emotional-Analysis
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3.**Run the Notebook**:
Open Emotions-analysis-gru.ipynb in Jupyter Notebook or Google Colab and run all cells.

---
## 🚀 Future Improvements
Add a Streamlit web interface for interactive predictions

Experiment with Bi-GRU or Attention mechanisms

Extend the dataset for multilingual or fine-grained emotion detection

---

## 🙋‍♂️ Author
Durgaprasad Nekkanti

---

## 🙏 Acknowledgement
This project was inspired by publicly available research and notebooks from the machine learning community. I used them as a learning foundation and built my implementation, structure, and analysis around it to understand GRU-based NLP classification more deeply.

---
## 🤝 Disclaimer
This project was developed for academic and learning purposes. The implementation is based on open-source ideas with modifications and personal enhancements.
