# 🚀 Sentiment Analysis on Social Media Posts

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive deep learning project that analyzes sentiment in social media posts using advanced neural network architectures including RNN, LSTM, and GRU models.

## 🎯 Project Overview

This project implements a robust sentiment analysis system capable of classifying social media posts as positive or negative. The system leverages multiple deep learning architectures and provides comparative analysis to determine the most effective approach for sentiment classification.

### ✨ Key Features

- 🧠 **Multiple Neural Network Architectures**: RNN, LSTM, GRU, and Bidirectional LSTM
- 📊 **Comprehensive Data Preprocessing**: Text cleaning, tokenization, and normalization
- 🎛️ **Advanced Model Features**: Layer normalization, dropout regularization, early stopping
- 📈 **Performance Visualization**: Training/validation accuracy and loss plots
- 🔍 **Real-time Prediction System**: Interactive sentiment detection with confidence scores
- 💾 **Model Persistence**: Save and load trained models for deployment

## 📊 Dataset

The project uses the **Amazon Reviews Dataset**, processing:
- **Training Data**: 12,000 reviews
- **Test Data**: 2,500 reviews
- **Classes**: Binary classification (Positive/Negative)

## 🏗️ Architecture Comparison

| Model | Architecture | Features |
|-------|-------------|----------|
| **RNN** | Simple RNN with normalization | Basic sequential processing |
| **LSTM** | Bidirectional LSTM | Long-term memory, bidirectional context |
| **GRU** | Gated Recurrent Units | Simplified LSTM with faster training |

All models include:
- 300-dimensional embedding layer
- Layer normalization for training stability
- Dropout layers for regularization
- Dense layers with ReLU activation
- Binary classification output

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-social-media.git
cd sentiment-analysis-social-media
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
```

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
nltk>=3.7
jupyter>=1.0.0
```

## 🚀 Usage

### Running the Complete Analysis

1. **Open Jupyter Notebook**
```bash
jupyter notebook sentiment-analysis-on-social-media-post.ipynb
```

2. **Run all cells sequentially** to:
   - Load and preprocess the data
   - Train multiple models
   - Compare performance metrics
   - Test the prediction system

### Quick Prediction

```python
from tensorflow.keras.models import load_model
import pickle

# Load trained model and tokenizer
model = load_model('lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict sentiment
def predict_sentiment(text):
    # Preprocessing and prediction logic
    processed_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)[0][0]
    
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😡"
    return sentiment, float(prediction)

# Example usage
text = "I love this product! It's amazing."
sentiment, confidence = predict_sentiment(text)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
```

## 📈 Model Performance

### Training Metrics
- **Accuracy**: Achieved high accuracy across all models
- **Loss Optimization**: Effective convergence with early stopping
- **Validation**: Robust performance on unseen data

### Key Preprocessing Steps
1. **Text Cleaning**: Lowercase conversion, special character removal
2. **Stopword Removal**: Filtering common English stopwords
3. **Stemming**: Reducing words to their root forms
4. **Tokenization**: Converting text to numerical sequences
5. **Padding**: Standardizing sequence lengths

## 🎨 Visualization

The project includes comprehensive visualizations:
- Training vs Validation Accuracy plots
- Training vs Validation Loss curves
- Confusion matrices for model evaluation
- Classification reports with precision, recall, and F1-scores

## 📁 Project Structure

```
sentiment-analysis-social-media/
│
├── sentiment-analysis-on-social-media-post.ipynb  # Main notebook
├── lstm_model.h5                                  # Trained LSTM model
├── tokenizer.pkl                                  # Fitted tokenizer
├── README.md                                      # Project documentation
├── requirements.txt                               # Dependencies
└── assets/                                        # Images and plots
```

## 🔧 Technical Implementation

### Data Preprocessing Pipeline
```python
def clean_text(text):
    # Lowercase conversion
    # Remove special characters
    # Stopword removal
    # Porter stemming
    # Return cleaned text
```

### Model Architecture (LSTM Example)
```python
model = Sequential([
    Embedding(max_words, 300, input_length=100),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(128, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

## 🎯 Future Enhancements

- [ ] **Transformer Models**: Implementation of BERT/RoBERTa
- [ ] **Multi-class Classification**: Extend beyond binary sentiment
- [ ] **Real-time API**: Flask/FastAPI deployment
- [ ] **Web Interface**: Interactive web application
- [ ] **Social Media Integration**: Direct platform APIs
- [ ] **Emotion Detection**: Fine-grained emotion classification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**PRIYANSH DIXIT**
- GitHub: [@PriyanshDixit](https://github.com/PRIYANSH-DIXIT)
- Email: priyanshdixit27@gmail.com

## 🙏 Acknowledgments

- Amazon for providing the review dataset
- TensorFlow team for the deep learning framework
- NLTK contributors for text processing tools
- The open-source community for inspiration and support

## 📞 Support

If you have any questions or need support, please:
1. Check the [Issues](https://github.com/yourusername/sentiment-analysis-social-media/issues) page
2. Create a new issue with a detailed description
3. Contact me directly via email

---

⭐ **Star this repository if you found it helpful!**

*Made with ❤️ and Python*
