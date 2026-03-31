import os
import re
import pandas as pd
import logging
from .config import START_DATE, END_DATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def limit_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces the study's strict start and end dates.
    Dropping any records outside of this window.
    """
    if 'created_at' not in df.columns:
        logging.error("DataFrame must contain 'created_at' column.")
        return df
        
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    
    # Drop rows where dates are NaT (could not be parsed)
    initial_len = len(df)
    df = df.dropna(subset=['created_at'])
    
    # Filter bounds
    valid_window = (df['created_at'] >= START_DATE) & (df['created_at'] <= END_DATE)
    df_filtered = df[valid_window].copy()
    
    dropped = initial_len - len(df_filtered)
    logging.info(f"Time window filter: Dropped {dropped} tweets outside {START_DATE.date()} to {END_DATE.date()}")
    
    # Add useful date/week columns
    df_filtered['date'] = df_filtered['created_at'].dt.date
    # Calculate week relative to start date
    df_filtered['week'] = ((df_filtered['created_at'] - START_DATE).dt.days // 7).astype(int)
    
    return df_filtered

def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Basic text cleaning.
    """
    if not isinstance(text, str):
         return ""
         
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user tags
    text = re.sub(r'\@\w+', '', text)
    # Remove newlines
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if lowercase:
        text = text.lower()
        
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline:
    1. Time window constraint
    2. Retweet flagging
    3. Dual-track string cleaning
    """
    df = limit_time_window(df)
    
    if df.empty:
        logging.warning("DataFrame is empty after time window filtering.")
        return df
        
    # Handle retweets
    df['is_retweet'] = df['text'].str.startswith('RT ', na=False)
    
    # Dual-track preprocessing
    logging.info("Applying dual-track text cleaning...")
    
    # Track A: For VADER/NRC (lowercased)
    df['clean_text_lexicon'] = df['text'].map(lambda x: clean_text(x, lowercase=True))
    
    # Track B: For BERTweet (case preserved, but clean)
    df['clean_text_bert'] = df['text'].map(lambda x: clean_text(x, lowercase=False))
    
    return df

def process_file(input_csv: str, output_csv: str):
    """
    Utility wrapper to read, clean, and save a batch.
    """
    try:
        df = pd.read_csv(input_csv)
        df_clean = preprocess_dataframe(df)
        df_clean.to_csv(output_csv, index=False)
        logging.info(f"Processed {len(df_clean)} records to {output_csv}")
    except Exception as e:
        logging.error(f"Error processing {input_csv}: {str(e)}")
