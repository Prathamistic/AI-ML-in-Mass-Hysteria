import os
import pandas as pd
import logging
from .config import START_DATE, END_DATE, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_daily_sequence(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Constructs a robust daily multivariate sequence with exactly 7 features.
    No imputation handles missing data (they remain NaN, handled by model).
    Adds a low_confidence flag based on volume.
    """
    if 'date' not in df.columns:
        logging.error("DataFrame must contain 'date' column for aggregation.")
        return df
        
    logging.info(f"Building daily aggregate sequence from {START_DATE.date()} to {END_DATE.date()}...")
    
    # Expected output features needed
    # mean_sentiment, pct_negative, mean_distress, rumor_velocity,
    # volume, retweet_ratio, crisis_kw_density
    
    # 1. Group by specific date
    daily = df.groupby('date').agg(
        volume=('tweet_id', 'count'),
        mean_sentiment=('vader_compound', 'mean'),
        sum_negative=('vader_neg', 'sum'),
        mean_distress=('nrc_distress', 'mean'),
        sum_rumors=('rumor_flag', 'sum'),
        sum_retweets=('is_retweet', 'sum'),
        sum_crisis_kws=('crisis_flag', 'sum')
    )
    
    # 2. Derive ratios / densities
    # Pct_negative: Amount of negative sentiment weighted by volume
    daily['pct_negative'] = daily['sum_negative'] / daily['volume']
    
    # Rumor_velocity: Sum of rumors relative to daily volume
    daily['rumor_velocity'] = daily['sum_rumors'] / daily['volume']
    
    # Retweet_ratio: Proportion of tweets that are retweets
    daily['retweet_ratio'] = daily['sum_retweets'] / daily['volume']
    
    # CrisisKeywordDensity: Proportion of tweets containing crisis terms
    daily['crisis_kw_density'] = daily['sum_crisis_kws'] / daily['volume']
    
    # Drop intermediate sums
    daily = daily.drop(columns=['sum_negative', 'sum_rumors', 'sum_retweets', 'sum_crisis_kws'])
    
    # 3. Time Index Re-alignment
    # Create complete date range to ensure no missing gaps are just skipped
    full_range = pd.date_range(start=START_DATE, end=END_DATE, tz=None).date
    
    daily.index = pd.to_datetime(daily.index).date
    daily = daily.reindex(full_range)
    daily.index.name = 'date'
    
    # 4. Low Confidence Flag
    # Hardcoded threshold, if less than X tweets in a day, mark low confidence
    # We use 50 here as a stand-in, but can be scaled if hydrating less
    daily['low_confidence'] = (daily['volume'] < 50) | (daily['volume'].isna())
    
    # Create directory if doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    daily.to_csv(output_file)
    logging.info(f"Aggregate sequence saved to {output_file} with shape {daily.shape}")
    
    return daily
