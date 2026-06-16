import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Wrapper for evaluation dataset
class EvalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_model():
    csv_path = "WELFake_Dataset.csv"
    model_path = "local_model"
    
    print("-" * 60)
    print("         Model Accuracy Evaluation on Complete WELFake Dataset")
    print("-" * 60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train the model first.")
        return
        
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at '{csv_path}'.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU for evaluation: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for evaluation (this may take a few minutes)...")
        
    # Load and preprocess dataset
    print("\nLoading dataset (Complete 72,134 Articles)...")
    df = pd.read_csv(csv_path)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    
    # Same pre-processing mix used during training to keep metrics aligned
    np.random.seed(42)
    mask = (np.random.rand(len(df)) < 0.5) & (df["title"].str.strip() != "")
    df["input_text"] = df["title"] + " " + df["text"]
    df.loc[mask, "input_text"] = df.loc[mask, "title"]
    
    df = df[["input_text", "label"]]
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    eval_texts = df["input_text"].tolist()
    eval_labels = df["label"].tolist()
    
    print(f"Total evaluation set size: {len(eval_labels)} samples (Complete Dataset).")
    
    print("Loading model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Tokenizing entire dataset...")
    encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=256)
    dataset = EvalDataset(encodings, eval_labels)
    
    # Use DataLoader to batch predictions
    batch_size = 32 if device == "cuda" else 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    print("Running batch predictions (Evaluating all 72,134 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
            # Print simple console progress bar
            progress = (i + 1) / len(dataloader)
            filled = int(progress * 20)
            bar = "=" * filled + " " * (20 - filled)
            sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%")
            sys.stdout.flush()
            
    print("\nCalculation complete. Compiling statistics...")
    
    # Calculate metrics manually
    tp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1)
    
    total = len(all_targets)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("                 MODEL VALIDATION REPORT (COMPLETE DATASET)")
    print("==========================================================")
    print(f"Accuracy:  {GREEN}{accuracy * 100:.2f}%{RESET}")
    print(f"F1 Score:  {CYAN}{f1 * 100:.2f}%{RESET}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(f"                  Predicted Real      Predicted Fake")
    print(f"Actual Real (0):       {tn:<10}          {fp:<10}")
    print(f"Actual Fake (1):       {fn:<10}          {tp:<10}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_model()
