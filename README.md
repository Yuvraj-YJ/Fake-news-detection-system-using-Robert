# 🔍 Fake News Detection System Using RoBERTa

A high-performance, GPU-accelerated Natural Language Processing (NLP) system designed to detect and classify misinformation. Powered by a fine-tuned **RoBERTa-base** transformer model, the system classifies news articles and headlines as **Real** or **Fake** with **97.42% accuracy**.

The application runs locally and supports both **manual text entry** and **real-time speech-to-text input** via microphone.

---

## ✨ Key Features

* **Advanced NLP Model:** Powered by a fine-tuned `roberta-base` model (125M parameters) trained on the academic benchmark **WELFake** dataset.
* **Dual Input Modes:**
  * **Text Input:** Paste or type news articles/headlines directly.
  * **Voice Input:** Record speech via a microphone, automatically transcribed using Google Web Speech API.
* **GPU Acceleration:** Fully optimized for NVIDIA CUDA GPU acceleration, enabling real-time inference (<0.05 seconds). Fallback to CPU is fully supported.
* **Robust Preprocessing:** Automated text formatting and capitalization restoration to ensure high prediction accuracy for speech transcriptions.
* **Audit Logging:** Every prediction is logged to `prediction_history.csv` with timestamps, input mode, text, verdict, and confidence score.

---

## 📈 Model Performance Metrics

The fine-tuned model achieved the following performance metrics on the complete **WELFake dataset** (72,134 news articles):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.42%** |
| **F1-Score** | **97.48%** |
| **Precision** | **98.17%** |
| **Recall** | **96.79%** |

### Confusion Matrix
* **True Real:** 34,389 | **False Fake:** 639
* **False Real:** 1,224 | **True Fake:** 35,882

---

## 🏗️ Project Structure

```text
Fake news detection/
├── cli_app.py              # Main interactive CLI application (text + voice)
├── train.py                # Local training script for model fine-tuning
├── train_colab.py          # Google Colab notebook training script
├── evaluate.py             # Evaluation script for full dataset metrics
├── run_cli.bat             # One-click launcher for the CLI app (Windows)
├── run_evaluate.bat        # One-click launcher for evaluation
├── run_train.bat           # One-click launcher for local training
├── requirements.txt        # Python dependency specifications
├── examiner_guide.txt      # Quick testing guide with copy-paste examples
├── README.md               # Project documentation
└── prediction_history.csv  # Auto-generated prediction history log
```

---

## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.10+**
* **NVIDIA GPU** with CUDA support (optional, but highly recommended)
* **Microphone** (required for voice input mode)
* **Internet Connection** (for initial model downloading and voice transcription)

### Quick Start (Windows)
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Yuvraj-YJ/Fake-news-detection-system-using-Robert.git
   cd Fake-news-detection-system-using-Robert
   ```
2. **Download the Dataset:**
   * Download `WELFake_Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place it in the project root directory.
3. **Launch the CLI:**
   * Double-click **`run_cli.bat`**. This script automatically sets up the Python virtual environment (`venv/`), installs all required dependencies, checks for CUDA GPU capability, and boots the CLI application.

### Manual Linux / macOS Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run CLI application
python cli_app.py
```

---

## 🚀 Usage Guide

### 1. Manual Text Entry
1. Select option `[1]` in the CLI.
2. Paste or type the news article text/headline.
3. Press `Enter` twice to run the prediction.

### 2. Voice Input via Microphone
1. Select option `[2]` in the CLI.
2. Enter the recording duration (default: 8 seconds) and speak after the prompt.
3. Verify the transcribed text, and press `Y` to execute the prediction.

### 3. Model Evaluation
* Double-click `run_evaluate.bat` or run `python evaluate.py` to evaluate the model performance and view classification metrics.

---

## 📄 License & References

* **Model Architecture:** Based on Meta AI's [RoBERTa](https://arxiv.org/abs/1907.11692).
* **Dataset:** [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) (Kaggle).
* This project is intended for educational and academic demonstration purposes.
