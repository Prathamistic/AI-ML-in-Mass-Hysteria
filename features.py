import re
import pandas as pd
import numpy as np
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from .config import CRISIS_KEYWORDS, RUMOR_KEYWORDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize once globally
vader_analyzer = SentimentIntensityAnalyzer()

def compute_vader_scores(text: str) -> dict:
    """Returns VADER compound score."""
    if not isinstance(text, str):
         return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    return vader_analyzer.polarity_scores(text)

def compute_nrc_distress(text: str) -> float:
    """
    Computes a distress score [0, 1] using NRC emotion lexicon.
    Focuses on: fear, anger, sadness, disgust.
    """
    if not isinstance(text, str):
         return 0.0
         
    emotion = NRCLex(text)
    scores = emotion.raw_emotion_scores
    total = sum(scores.values())
    
    if total == 0:
        return 0.0
        
    distress = scores.get('fear', 0) + scores.get('anger', 0) + scores.get('sadness', 0) + scores.get('disgust', 0)
    return float(distress) / float(total)

def score_keywords(text: str, keywords: list) -> int:
    """
    Returns boolean or count of how many keywords from list are found.
    """
    if not isinstance(text, str):
         return 0
    words = text.split()
    count = sum(1 for w in words if any(kw in w for kw in keywords))
    return int(count > 0)

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with 'clean_text_lexicon', compute all NLP features.
    """
    if 'clean_text_lexicon' not in df.columns:
        logging.error("DataFrame must contain 'clean_text_lexicon' column.")
        return df
        
    logging.info(f"Computing NLP features for {len(df)} records...")
    
    # 1. Sentiment richness
    vader_res = df['clean_text_lexicon'].apply(compute_vader_scores).apply(pd.Series)
    df['vader_compound'] = vader_res['compound']
    df['vader_pos'] = vader_res['pos']
    df['vader_neu'] = vader_res['neu']
    df['vader_neg'] = vader_res['neg']
    
    # 3-class sentiment replacing binary mapping
    df['sentiment'] = 'neutral'
    df.loc[df['vader_compound'] > 0.05, 'sentiment'] = 'positive'
    df.loc[df['vader_compound'] < -0.05, 'sentiment'] = 'negative'
    
    # Provide binary for compatibility layer if needed but 3-class is primary
    df['is_negative'] = (df['sentiment'] == 'negative').astype(int)
    
    # 2. Distress & Rumor module
    df['nrc_distress'] = df['clean_text_lexicon'].apply(compute_nrc_distress)
    
    # Graceful fallback: If NRC fails (all zeroes), use normalized VADER neg
    fallback_mask = (df['nrc_distress'] == 0.0) & (df['vader_neg'] > 0)
    df.loc[fallback_mask, 'nrc_distress'] = df.loc[fallback_mask, 'vader_neg']
    
    # Keyword flags
    df['crisis_flag'] = df['clean_text_lexicon'].apply(lambda x: score_keywords(x, CRISIS_KEYWORDS))
    df['rumor_flag'] = df['clean_text_lexicon'].apply(lambda x: score_keywords(x, RUMOR_KEYWORDS))
    
    return df
