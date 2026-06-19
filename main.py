import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
import json
import os
from datasets import DatasetDict, concatenate_datasets
from transformers import Wav2Vec2FeatureExtractor
from utils import preprocess_function, collate_fn
from models import model_mtkd, model_kd, model_ft
from data import iemocap, fesc, cafe
from pipeline import pipeline_mtkd, pipeline_kd, pipeline_ft
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_EPOCHS", help="Number of Epochs", type=int, default=10)
    parser.add_argument("--LEARNING_RATE", help="Learning Rate", type=float, default=1e-5, choices=[1e-5, 3e-5, 5e-5])
    parser.add_argument("--BATCH_SIZE", help="Batch Size", type=int, default=8)
    parser.add_argument("--SESSION", help="Session/Split Number", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--TRAINING", help="1: Yes, 0: No", type=int, default=0, choices=[0, 1])
    parser.add_argument("--PARADIGM", help="Choose a Training Paradigm: FT, KD, or MTKD", type=str, default="MTKD", choices=["FT", "KD", "MTKD"])
    parser.add_argument("--LANGUAGE", help="Choose a Language: English (EN), Finnish (FI), or French (FR)", type=str, default="EN", choices=["EN", "FI", "FR"])
    parser.add_argument("--LINGUALITY", help="Choose the Linguality: Monolingual or Multilingual", type=str, default="Monolingual", choices=["Monolingual", "Multilingual"])
    parser.add_argument("--teacher_selection", help="Ablation: teacher weighting mechanism", type=str, default="cosine", choices=["cosine", "attention"])
    parser.add_argument("--config_path", help="Path to config file", type=str, default="config.json")
    parser.add_argument("--ce_weight", help="Weight for cross entropy loss", type=float, default=None)
    parser.add_argument("--kd_weight", help="Weight for knowledge distillation loss", type=float, default=None)
    parser.add_argument("--contrastive_weight", help="Weight for supervised contrastive loss", type=float, default=None)
    parser.add_argument("--contrastive_temp", help="Temperature for contrastive loss", type=float, default=None)
    parser.add_argument("--attention_hidden_dim", help="Hidden dimension for AttentionMTKD network", type=int, default=None)
    args = parser.parse_args()

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    N_EPOCHS = int(args.N_EPOCHS)
    LEARNING_RATE = float(args.LEARNING_RATE)
    BATCH_SIZE = int(args.BATCH_SIZE)
    SESSION = int(args.SESSION)
    TRAINING = int(args.TRAINING)
    PARADIGM = str(args.PARADIGM)
    LANGUAGE = str(args.LANGUAGE)
    LINGUALITY = str(args.LINGUALITY)

    if LANGUAGE == "EN" and SESSION > 5:
        raise ValueError("Error: English dataset IEMOCAP does not have more than five splits!")
    elif LANGUAGE == "FR" and SESSION > 1:
        raise ValueError("Error: French dataset CaFE does not have more than five splits!")

    if  LINGUALITY == "Monolingual":
        if LANGUAGE == "EN":
            label2id, id2label, ds = iemocap(SESSION)
        elif LANGUAGE == "FI":
            label2id, id2label, ds = fesc(SESSION)
        else: # LANGUAGE == "FR"
            label2id, id2label, ds = cafe(SESSION)

    elif LINGUALITY == "Multilingual":
        if LANGUAGE == "EN":
            label2id, id2label, ds_en = iemocap(session=SESSION)    # split i of en
            _, _, ds_fi = fesc(session=9)                           #    + best split (JAKA:9) from fi
            _, _, ds_fr = cafe(session=1)                           #    + only split from fr
        elif LANGUAGE == "FI":
            label2id, id2label, ds_fi = fesc(session=SESSION)       # split i of fi
            _, _, ds_en = iemocap(session=2)                        #    + best split (s2) from en
            _, _, ds_fr = cafe(session=1)                           #    + only split from fr
        else: # LANGUAGE == "FR":
            label2id, id2label, ds_fr = cafe(session=SESSION)       # only split from fr
            _, _, ds_en = iemocap(session=2)                        #    + best split (s2) from en
            _, _, ds_fi = fesc(session=9)                           #    + best split (JAKA:9) from fi
        
        # Concatenate train, test, and dev datasets across iemocap, fesc, and cafe
        train_ds = concatenate_datasets([ds_en['train'], ds_fi['train'], ds_fr['train']])
        test_ds = concatenate_datasets([ds_en['test'], ds_fi['test'], ds_fr['test']])
        dev_ds = concatenate_datasets([ds_en['dev'], ds_fi['dev'], ds_fr['dev']])

        ds = DatasetDict({
            'train': train_ds,
            'test': test_ds,
            'dev': dev_ds
        })

    print(ds)
    print(label2id)
    print(id2label)

    MODEL_CKPT = "facebook/wav2vec2-base"
    NUM_OF_LABELS = len(label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)

    encoded_audio = ds.map(lambda example: preprocess_function(example, feature_extractor), remove_columns="audio", batched=True)

    train_loader = DataLoader(encoded_audio["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(encoded_audio["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn)
    valid_loader = DataLoader(encoded_audio["dev"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Load config file if it exists
    config_data = {}
    if os.path.exists(args.config_path):
        try:
            with open(args.config_path, 'r') as f:
                config_data = json.load(f)
            print(f"Successfully loaded config from {args.config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    # Resolve loss weights and hyperparameters (CLI arguments override config file)
    ce_weight = args.ce_weight if args.ce_weight is not None else config_data.get("ce_weight", 0.75)
    kd_weight = args.kd_weight if args.kd_weight is not None else config_data.get("kd_weight", 0.25)
    contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else config_data.get("contrastive_weight", 0.1)
    contrastive_temp = args.contrastive_temp if args.contrastive_temp is not None else config_data.get("contrastive_temperature", 0.07)
    attention_hidden_dim = args.attention_hidden_dim if args.attention_hidden_dim is not None else config_data.get("attention_hidden_dim", 16)

    hyperparam = {
        "SESSION": SESSION,
        "LEARNING_RATE": LEARNING_RATE,
        "TRAINING": TRAINING,
        "N_EPOCHS": N_EPOCHS,
        "LINGUALITY": LINGUALITY,
        "LANGUAGE": LANGUAGE,
        "teacher_selection": args.teacher_selection,
        "ce_weight": ce_weight,
        "kd_weight": kd_weight,
        "contrastive_weight": contrastive_weight,
        "contrastive_temp": contrastive_temp,
        "attention_hidden_dim": attention_hidden_dim
    }

    if PARADIGM == "MTKD":
        teacher_en, teacher_fi, teacher_fr, student = model_mtkd(label2id, id2label, NUM_OF_LABELS, device)
        pipeline_mtkd(teacher_en, teacher_fi, teacher_fr, student, train_loader, valid_loader, test_loader, hyperparam, device)

    elif PARADIGM == "KD":
        teacher, student = model_kd(label2id, id2label, NUM_OF_LABELS, device)
        pipeline_kd(teacher, student, train_loader, valid_loader, test_loader, hyperparam, device)

    else: # PARADIGM == "FT"
        model = model_ft(label2id, id2label, NUM_OF_LABELS, device)
        pipeline_ft(model, train_loader, valid_loader, test_loader, hyperparam, device)
        

if __name__ == "__main__":
    main()
