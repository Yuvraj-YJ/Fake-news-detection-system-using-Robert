import os
import sys
import time
import warnings
import csv
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

if sys.platform == "win32":
    os.system("color")

classifier = None

def load_model():
    global classifier
    model_path = "local_model"
    
    print("\nLoading classification model...")
    if os.path.exists(model_path):
        print(f"Loading local fine-tuned model from '{model_path}'...")
        model_source = model_path
    else:
        print("Local model not found. Loading online model (hamzab/roberta-fake-news-classification)...")
        model_source = "hamzab/roberta-fake-news-classification"
        
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        if device_id == 0:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for inference.")
            
        classifier = pipeline(
            "text-classification",
            model=model_source,
            device=device_id
        )
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load fallback model...")
        try:
            classifier = pipeline("text-classification", model="Arko007/fake-news-roberta-5M")
            print("Fallback model loaded successfully.\n")
        except Exception as e2:
            print(f"Fatal error: Could not load fallback model: {e2}")
            sys.exit(1)

def record_voice(duration=8, samplerate=44100):
    filename = "temp_voice.wav"
    
    print("\nGet ready to speak...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
        
    print("\nRecording started... speak now.")
    
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        
        # Simple progress bar
        bar_len = 25
        for t in range(duration * 2):
            progress = (t + 1) / (duration * 2)
            filled = int(progress * bar_len)
            bar = "=" * filled + " " * (bar_len - filled)
            sys.stdout.write(f"\rRecording: [{bar}] {int(progress * 100)}%")
            sys.stdout.flush()
            time.sleep(0.5)
            
        sd.wait()
        print("\nRecording finished.")
        sf.write(filename, recording, samplerate)
        return filename
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def transcribe(filename):
    if not filename or not os.path.exists(filename):
        return None
        
    print("Transcribing voice input...")
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"Transcribed: \"{text}\"")
            return text
    except sr.UnknownValueError:
        print("Warning: Speech was not clear enough to recognize.")
        return None
    except sr.RequestError as e:
        print(f"Google speech recognition service error: {e}")
        return None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

def predict(text, mode):
    if not classifier:
        print("Error: Model is not loaded.")
        return
        
    text = text.strip()
    if not text:
        print("Error: Empty input.")
        return

    # Auto-capitalize the first letter to match formal news headline style.
    # This prevents lowercase voice transcriptions from being misclassified due to tokenizer case-sensitivity.
    if len(text) > 0 and text[0].islower():
        text = text[0].upper() + text[1:]

    print("Analyzing text...")
    
    # Truncate to match model length limits
    truncated = text[:3000]
    
    try:
        res = classifier(truncated, truncation=True, max_length=512)[0]
        raw_label = res["label"]
        raw_score = res["score"]
        
        label_lower = raw_label.lower()
        if "fake" in label_lower or "label_1" in label_lower or "false" in label_lower:
            verdict = "FAKE"
            color = RED
        else:
            verdict = "REAL"
            color = GREEN
            
        print("\n" + "-" * 50)
        print("                 DETECTION RESULTS")
        print("-" * 50)
        print(f"Input Mode: {mode}")
        print(f"Text:       \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"Words:      {len(text.split())}")
        print("-" * 50)
        print(f"Verdict:    {color}[{verdict}]{RESET}")
        print(f"Confidence: {color}{raw_score * 100:.2f}%{RESET}")
        print("-" * 50 + "\n")

        # Log prediction to a CSV file
        log_file = "prediction_history.csv"
        file_exists = os.path.exists(log_file)
        try:
            with open(log_file, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Input Mode", "Text", "Verdict", "Confidence"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    mode,
                    text,
                    verdict,
                    f"{raw_score * 100:.2f}%"
                ])
            print(f"{CYAN}Prediction logged to '{log_file}'{RESET}\n")
        except Exception:
            pass
        
    except Exception as e:
        print(f"Analysis failed: {e}")

def main():
    load_model()
    
    while True:
        print("==================================================")
        print("          Fake News Detection CLI")
        print("==================================================")
        print("[1] Paste news text manually")
        print("[2] Input news text via microphone")
        print("[3] Exit")
        print("-" * 50)
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            print("\nEnter the news content below (Press Enter twice to analyze):")
            print(f"{YELLOW}(Tip: Enter formal statements like 'Trump is the president' instead of questions like 'is Trump the president'){RESET}")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            
            text = " ".join(lines).strip()
            if text:
                predict(text, "Manual Entry")
            else:
                print("No text entered.\n")
                
        elif choice == "2":
            print(f"\n{YELLOW}(Tip: Speak formal statements instead of questions for best classification){RESET}")
            dur_val = input("Enter voice recording duration in seconds (default 8s): ").strip()
            duration = 8
            if dur_val.isdigit():
                duration = int(dur_val)
                
            audio_file = record_voice(duration=duration)
            if audio_file:
                txt = transcribe(audio_file)
                if txt:
                    run_check = input("\nRun detection on this statement? (Y/n): ").strip().lower()
                    if run_check in ["", "y", "yes"]:
                        predict(txt, "Voice Input")
                    else:
                        print("Cancelled.\n")
                else:
                    print("Could not transcribe audio.\n")
                    
        elif choice == "3":
            print("Exiting. Thank you.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 1, 2, or 3.\n")

if __name__ == "__main__":
    main()