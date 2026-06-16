import os
import sys
import torch
import pandas as pd
import zipfile
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# PyTorch wrapper for dataset
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    csv_path = "WELFake_Dataset.csv"
    output_dir = "local_model"
    zip_path = "local_model.zip"
    
    print("-" * 60)
    print("         Google Colab Model training Script")
    print("-" * 60)
    
    # Locate dataset file
    if not os.path.exists(csv_path):
        print(f"'{csv_path}' not found in current folder. Checking Google Drive mount...")
        drive_path = "/content/drive/MyDrive/WELFake_Dataset.csv"
        if os.path.exists(drive_path):
            print(f"Found dataset in Google Drive: {drive_path}")
            csv_path = drive_path
        else:
            print("\nError: Please upload 'WELFake_Dataset.csv' to Google Colab or Drive.")
            print("Mount Drive using:")
            print("  from google.colab import drive")
            print("  drive.mount('/content/drive')")
            return

    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    
    # Mix title-only (short) and title + text (long) inputs (50% each)
    import numpy as np
    np.random.seed(42)
    mask = (np.random.rand(len(df)) < 0.5) & (df["title"].str.strip() != "")
    
    df["input_text"] = df["title"] + " " + df["text"]
    df.loc[mask, "input_text"] = df.loc[mask, "title"]
    
    df = df[["input_text", "label"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Hardware check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")
    
    # Train/validation split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_texts = train_df["input_text"].tolist()
    train_labels = train_df["label"].tolist()
    
    val_texts = val_df["input_text"].tolist()
    val_labels = val_df["label"].tolist()
    
    # Tokenize input data
    print("Tokenizing texts using roberta-base tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
    
    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    val_dataset = FakeNewsDataset(val_encodings, val_labels)
    
    # Load model architecture
    print("Loading roberta-base model structure...")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)
    
    # Training configuration parameters
    print("Setting up training configurations...")
    use_cpu_flag = (device == "cpu")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,              
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",      
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        disable_tqdm=False,
        use_cpu=use_cpu_flag
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training process...")
    trainer.train()
    
    print(f"Training completed successfully. Saving model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model files successfully saved.")
    
    # Zip output directory for easy downloads
    print(f"Zipping model files into '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.join(output_dir, '..')))
    print(f"Zip file created successfully at: {zip_path}")
    print("Done. You can now download the zip file from Colab.")

if __name__ == "__main__":
    train_model()
