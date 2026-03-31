import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .config import BERTWEET_MODEL, BATCH_SIZE, EMBEDDINGS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Can add MPS for Apple Silicon later if requested
    return torch.device('cpu')

def load_bertweet_model():
    """Loads tokenizer and model to available device."""
    logging.info(f"Loading {BERTWEET_MODEL}...")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL)
    model = AutoModel.from_pretrained(BERTWEET_MODEL).to(device)
    model.eval()
    return tokenizer, model, device

def generate_embeddings(df: pd.DataFrame, file_prefix: str = "chunk"):
    """
    Takes a dataframe with 'clean_text_bert' and produces CLS embeddings.
    Caches results to disk to avoid recomputing.
    """
    if 'clean_text_bert' not in df.columns:
        logging.error("DataFrame must contain 'clean_text_bert' column.")
        return False
        
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    tokenizer, model, device = load_bertweet_model()
    
    # Track indices for mapping back to CSV
    texts = df['clean_text_bert'].tolist()
    total_samples = len(texts)
    
    embeddings = []
    
    cache_file = os.path.join(EMBEDDINGS_DIR, f"{file_prefix}_embeddings.npy")
    
    if os.path.exists(cache_file):
         logging.info(f"Cache hit: Loading embeddings from {cache_file}")
         return True # Already cached
         
    logging.info(f"Generating embeddings for {total_samples} texts on {device}...")
    
    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Embedding Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Use mixed precision if on GPU
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
                
            # Extract CLS token from last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    
    # Save to disk
    np.save(cache_file, all_embeddings)
    logging.info(f"Saved {all_embeddings.shape} embeddings to {cache_file}")
    
    return True
