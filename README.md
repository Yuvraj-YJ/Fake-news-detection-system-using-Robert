# 🔍 Fake News Detection System Using RoBERTa

A GPU-accelerated, command-line fake news detection system powered by a fine-tuned **RoBERTa** (Robustly Optimized BERT Pretraining Approach) transformer model. The system classifies news headlines and articles as **Real** or **Fake** using both manual text input and real-time **Speech-to-Text** voice input.

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Algorithm Selection & Comparison](#algorithm-selection--comparison)
- [Model Performance Metrics](#model-performance-metrics)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Testing Guidelines](#testing-guidelines)
- [Session Logging](#session-logging)
- [License](#license)

---

## 📝 Abstract

The proliferation of fake news on digital platforms has become a critical global challenge, threatening democratic processes, public health, and social cohesion. This project presents a **local, GPU-accelerated Fake News Detection System** that leverages the power of the **RoBERTa (Robustly Optimized BERT Pretraining Approach)** transformer model, fine-tuned on the **WELFake dataset** containing **72,134 labeled news articles**.

The system operates as a lightweight **Command-Line Interface (CLI)** application that supports two input modes:
1. **Manual Text Entry** — Users paste or type news headlines and articles directly.
2. **Voice Input via Microphone** — Real-time speech is captured, transcribed using Google's Speech Recognition API, and then classified.

The fine-tuned RoBERTa model achieves a classification accuracy of **97.42%**, an F1-Score of **97.48%**, precision of **98.17%**, and recall of **96.79%** on the complete dataset. The model operates as a **stylistic and linguistic classifier**, analyzing syntax, vocabulary patterns, sentence structure, and writing style to distinguish between credible journalistic reporting and unreliable clickbait or sensationalized content.

Key features include GPU-accelerated inference using NVIDIA CUDA, automatic input preprocessing (capitalization correction for speech transcriptions), and a session logging system that records all predictions to a CSV file for audit and review purposes.

**Keywords:** Fake News Detection, Natural Language Processing, RoBERTa, Transformer Models, Deep Learning, Text Classification, Speech-to-Text, CUDA, GPU Acceleration

---

## 🎯 Motivation

### Why Build This Project?

1. **Growing Misinformation Crisis:** Social media platforms have amplified the spread of false information at an unprecedented scale. During events like elections, health crises (COVID-19), and geopolitical conflicts, fake news has caused real-world harm including vaccine hesitancy, election interference, and communal violence.

2. **Limitations of Manual Fact-Checking:** Human fact-checkers cannot keep pace with the volume of content produced daily. Automated detection systems are essential to flag potentially misleading content in real-time.

3. **Need for Accessible Tools:** Most existing fake news detection systems are cloud-based APIs or complex research pipelines. This project provides a **simple, local, offline-capable** tool that runs on consumer hardware (an NVIDIA GPU or CPU), making advanced NLP accessible without cloud dependencies.

4. **Stylistic Analysis Approach:** Rather than relying on external knowledge bases or fact databases, this system analyzes the **writing style** of news content — detecting linguistic patterns associated with clickbait, sensationalism, and unreliable sources versus professional journalism.

5. **Speech-to-Text Integration:** Adding voice input capabilities makes the system more versatile and accessible, enabling users to simply speak a headline they heard on television, radio, or in conversation and receive an instant classification.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│                                                         │
│    [Option 1]              [Option 2]                   │
│    Manual Text             Microphone                   │
│    Entry                   Recording                    │
│       │                       │                         │
│       │                       ▼                         │
│       │               ┌──────────────┐                  │
│       │               │  sounddevice  │                  │
│       │               │  (WAV Record) │                  │
│       │               └──────┬───────┘                  │
│       │                      ▼                          │
│       │               ┌──────────────┐                  │
│       │               │   Google      │                  │
│       │               │ Speech API    │                  │
│       │               │ (Transcribe)  │                  │
│       │               └──────┬───────┘                  │
│       │                      │                          │
│       ▼                      ▼                          │
│  ┌────────────────────────────────────┐                 │
│  │      Auto-Capitalize First Letter  │                 │
│  │      (Preprocessing Step)          │                 │
│  └──────────────┬─────────────────────┘                 │
│                 ▼                                       │
│  ┌────────────────────────────────────┐                 │
│  │     RoBERTa Tokenizer              │                 │
│  │     (roberta-base vocabulary)      │                 │
│  └──────────────┬─────────────────────┘                 │
│                 ▼                                       │
│  ┌────────────────────────────────────┐                 │
│  │   Fine-Tuned RoBERTa Model         │                 │
│  │   (125M Parameters)                │                 │
│  │   GPU: NVIDIA CUDA Acceleration    │                 │
│  └──────────────┬─────────────────────┘                 │
│                 ▼                                       │
│  ┌────────────────────────────────────┐                 │
│  │     Classification Output          │                 │
│  │     REAL / FAKE + Confidence %     │                 │
│  └──────────────┬─────────────────────┘                 │
│                 ▼                                       │
│  ┌────────────────────────────────────┐                 │
│  │   Session Logger                   │                 │
│  │   (prediction_history.csv)         │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### WELFake Dataset

| Property | Details |
|----------|---------|
| **Name** | WELFake (WELl-known Fake news dataset) |
| **Total Articles** | 72,134 |
| **Real News Articles** | 35,028 |
| **Fake News Articles** | 37,106 |
| **Sources** | Kaggle, McIntire, Reuters, BuzzFeed Political |
| **Columns** | `title` (headline), `text` (article body), `label` (0 = Real, 1 = Fake) |
| **Format** | CSV (≈245 MB) |
| **Academic Reference** | Peer-reviewed benchmark dataset widely used in fake news detection research |

### Data Preparation Strategy

A **50/50 mixed-length training strategy** was employed to ensure the model handles both short and long inputs effectively:

- **50% of samples** — Trained on **title only** (short headlines, typically 5–15 words)
- **50% of samples** — Trained on **title + full article text** (long-form content, 100–1000+ words)

This strategy prevents the model from developing a bias toward long-form text and ensures accurate classification when users input short voice-transcribed headlines.

### Train/Validation Split

| Split | Articles | Percentage |
|-------|----------|------------|
| **Training Set** | 57,707 | 80% |
| **Validation Set** | 14,427 | 20% |
| **Total** | 72,134 | 100% |

---

## 🧠 Algorithm Selection & Comparison

### Why RoBERTa?

**RoBERTa (Robustly Optimized BERT Pretraining Approach)** was selected as the core classification algorithm after evaluating multiple approaches. RoBERTa, developed by Meta AI (Facebook), is an enhanced version of BERT that was trained with larger batches, more data, longer sequences, and dynamic masking — resulting in superior performance on text classification benchmarks.

### Comparison with Alternative Approaches

| Algorithm | Type | Accuracy (Typical) | Strengths | Weaknesses | Chosen? |
|-----------|------|-------------------|-----------|------------|---------|
| **RoBERTa-base** | Transformer (125M params) | **97–98%** | State-of-the-art NLP; captures deep contextual meaning; handles both short & long text; fast GPU inference (<0.05s) | Requires GPU for optimal training speed; 500MB model size | ✅ **Yes** |
| Naive Bayes | Traditional ML | 70–80% | Simple, fast training, low resource usage | Cannot capture word order or context; poor on nuanced text | ❌ |
| Logistic Regression + TF-IDF | Traditional ML | 75–85% | Interpretable, fast | Bag-of-words approach ignores word relationships and sentence structure | ❌ |
| Random Forest + TF-IDF | Ensemble ML | 78–86% | Good for tabular features | Cannot understand semantic meaning of sentences | ❌ |
| LSTM / BiLSTM | Recurrent Neural Network | 85–92% | Captures sequential patterns in text | Slow training; struggles with very long text; vanishing gradient issues | ❌ |
| BERT-base | Transformer (110M params) | 94–96% | Strong contextual understanding | RoBERTa outperforms BERT due to optimized pretraining | ❌ |
| **RoBERTa-large** | Transformer (355M params) | 97–99% | Slightly higher potential accuracy | 3× more GPU memory required; OOM errors on consumer GPUs; marginal improvement | ❌ |
| DeBERTa-v3 | Transformer (184M params) | 97–98% | Highest benchmark scores | Slower inference; complex tokenizer; marginal gain over RoBERTa-base | ❌ |
| GPT-4 / LLMs | Large Language Model (175B+ params) | 95–98% | Massive reasoning capability | Requires cloud API; extremely slow; infeasible for local real-time CLI | ❌ |

### Key Reasons for Choosing RoBERTa-base

1. **Optimal Balance:** Achieves near state-of-the-art accuracy (97.42%) while running instantly on consumer GPUs.
2. **Real-Time Inference:** Prediction latency is under **0.05 seconds** on an NVIDIA GPU, critical for smooth interactive CLI use.
3. **Manageable Model Size:** At approximately **500 MB**, the model is easily stored and loaded locally without cloud dependency.
4. **Robust Pretraining:** RoBERTa's enhanced pretraining (dynamic masking, larger batches, removal of Next Sentence Prediction) gives it an edge over standard BERT.
5. **Academic Recognition:** RoBERTa is a widely cited, peer-reviewed model from Meta AI, lending academic credibility to the project.

---

## 📈 Model Performance Metrics

The following metrics were computed by evaluating the fine-tuned model on the **complete WELFake dataset (72,134 articles)**:

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.42%** |
| **F1-Score** | **97.48%** |
| **Precision** | **98.17%** |
| **Recall** | **96.79%** |

### Confusion Matrix

|  | Predicted Real | Predicted Fake |
|--|---------------|----------------|
| **Actual Real (0)** | 34,389 | 639 |
| **Actual Fake (1)** | 1,224 | 35,882 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | `roberta-base` (125M parameters) |
| **Training Epochs** | 2 |
| **Batch Size** | 8 |
| **Max Token Length** | 256 |
| **Warmup Steps** | 100 |
| **Weight Decay** | 0.01 |
| **Optimizer** | AdamW (default HuggingFace Trainer) |
| **Learning Rate** | 5e-5 (default) |
| **Evaluation Strategy** | End of each epoch |
| **Best Model Selection** | Lowest validation loss |

---

## 🛠️ Technology Stack

### Programming Language
| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language for the entire project |

### Deep Learning & NLP
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.x (CUDA 13.0) | Deep learning framework for model training and GPU-accelerated inference |
| **HuggingFace Transformers** | 4.x | Provides RoBERTa model architecture, tokenizer, and training utilities |
| **Accelerate** | Latest | Optimizes distributed and mixed-precision training |

### Speech Processing
| Library | Purpose |
|---------|---------|
| **SpeechRecognition** | Interfaces with Google Web Speech API for speech-to-text transcription |
| **sounddevice** | Records audio from the system microphone |
| **soundfile** | Reads and writes WAV audio files |

### Data Processing
| Library | Purpose |
|---------|---------|
| **Pandas** | Loads and preprocesses the CSV dataset |
| **NumPy** | Numerical operations and random seed management for reproducibility |

### Hardware
| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA RTX 5060 Ti (CUDA-accelerated training and inference) |
| **Fallback** | CPU mode supported (slower but functional) |

---

## 📁 Project Structure

```
Fake news detection/
│
├── cli_app.py                  # Main CLI application (text + voice input)
├── train.py                    # Local GPU/CPU training script
├── train_colab.py              # Google Colab training script (cloud GPU)
├── evaluate.py                 # Full dataset evaluation & metrics calculator
│
├── run_cli.bat                 # One-click launcher for CLI app (Windows)
├── run_train.bat               # One-click launcher for local training
├── run_evaluate.bat            # One-click launcher for model evaluation
│
├── requirements.txt            # Python dependencies
├── examiner_guide.txt          # Testing guide with copy-paste examples
├── README.md                   # This documentation file
│
├── WELFake_Dataset.csv         # Complete dataset (72,134 articles, ~245 MB)
│
├── local_model/                # Trained model weights & tokenizer
│   ├── config.json             # Model configuration
│   ├── model.safetensors       # Trained weight parameters (~500 MB)
│   ├── tokenizer.json          # Tokenizer vocabulary
│   └── tokenizer_config.json   # Tokenizer settings
│
├── prediction_history.csv      # Auto-generated session log of all predictions
├── results/                    # Training checkpoints (generated during training)
└── venv/                       # Python virtual environment (not committed)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.10** or higher
- **NVIDIA GPU** with CUDA support (recommended) or CPU (slower)
- **Microphone** (required only for voice input mode)
- **Internet connection** (required only for initial dependency download and voice transcription)

### Quick Start (Windows)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yuvraj-YJ/fake-news-detection-roberta.git
   cd fake-news-detection-roberta
   ```

2. **Download the dataset:**
   - Download `WELFake_Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place it in the project root directory.

3. **Run the application:**
   - Double-click **`run_cli.bat`** — this will:
     - Create a Python virtual environment (`venv/`)
     - Install all dependencies from `requirements.txt`
     - Auto-detect GPU/CPU
     - Launch the CLI application

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run CLI
python cli_app.py

# Run evaluation
python evaluate.py

# Train model (optional, pre-trained weights included)
python train.py
```

---

## 🚀 Usage Guide

### Option 1: Manual Text Entry
1. Launch the application (double-click `run_cli.bat` or run `python cli_app.py`)
2. Select **[1] Paste news text manually**
3. Type or paste a news headline/article
4. Press **Enter twice** to analyze
5. View the **REAL** or **FAKE** verdict with confidence score

### Option 2: Voice Input via Microphone
1. Launch the application
2. Select **[2] Input news text via microphone**
3. Set recording duration (default: 8 seconds)
4. Speak a news headline clearly after the countdown
5. Confirm the transcription
6. View the classification result

### Option 3: Evaluate Model Accuracy
- Double-click **`run_evaluate.bat`** to run the full dataset evaluation
- The script processes all 72,134 articles and displays accuracy, F1-score, precision, recall, and a confusion matrix

---

## 🧪 Testing Guidelines

### Input Style Recommendations

The model operates as a **stylistic and linguistic classifier**. For best results, input text should be formatted as **formal news statements** rather than casual questions:

| ✅ Correct Format | ❌ Incorrect Format |
|-------------------|---------------------|
| "Donald Trump is the president of the United States" | "is Trump the president of USA?" |
| "A study suggests regular exercise may improve cognitive performance" | "does exercise help the brain?" |
| "India defeated Australia in the Cricket World Cup final" | "who won the cricket world cup?" |

### Sample Test Cases

#### Real News (Expected: REAL)
```
A study suggests regular exercise may improve cognitive performance
```
```
Schumer calls on Trump to appoint official to oversee Puerto Rico relief
```
```
Billionaire Odebrecht in Brazil scandal released to house arrest
```

#### Fake / Clickbait News (Expected: FAKE)
```
BREAKING: OBAMA SPOTTED AT SECRET REPTILIAN MEETING IN AREA 51!!! [WATCH]
```
```
UNBELIEVABLE! HILLARY CLINTONS PRIVATE EMAIL SERVER DISCOVERED IN CONEY ISLAND PIZZERIA!!!
```
```
SATAN 2: Russia unveils an image of its terrifying new SUPERNUKE - Western world takes notice
```

---

## 📝 Session Logging

Every prediction made through the CLI is automatically logged to **`prediction_history.csv`** with the following fields:

| Column | Description |
|--------|-------------|
| **Timestamp** | Date and time of the prediction |
| **Input Mode** | Manual Entry or Voice Input |
| **Text** | The input text that was classified |
| **Verdict** | REAL or FAKE |
| **Confidence** | Model confidence percentage |

This file serves as an audit log and can be presented to examiners as evidence of live testing.

---

## 📄 License

This project is developed for academic and educational purposes.

- **Model:** Based on [RoBERTa](https://arxiv.org/abs/1907.11692) by Meta AI (Facebook), available under the MIT License.
- **Dataset:** [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) is publicly available on Kaggle for research use.

---

## 👤 Author

**Yuvraj**
- GitHub: [@Yuvraj-YJ](https://github.com/Yuvraj-YJ)
