import os
import time
import json
import logging
import subprocess
from glob import glob
from .config import TWITTER_BEARER_TOKEN, RAW_DIR, HYDRATED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hydrate_ids(id_file, output_file):
    if not TWITTER_BEARER_TOKEN:
        logging.error("Cannot hydrate without TWITTER_BEARER_TOKEN")
        return False
        
    logging.info(f"Hydrating {id_file} into {output_file}")
    
    # Using twarc2 for hydration, assumes twarc is installed and configured
    # We pass the bearer token as an environment variable
    env = os.environ.copy()
    env["BEARER_TOKEN"] = TWITTER_BEARER_TOKEN
    
    try:
        # Check if the file is empty first to avoid twarc errors
        with open(id_file, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                logging.warning(f"File {id_file} is empty, skipping.")
                return True
                
        cmd = ["twarc2", "hydrate", id_file, output_file]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error hydrating {id_file}: {result.stderr}")
            return False
            
        logging.info(f"Successfully hydrated {id_file}")
        return True
        
    except Exception as e:
        logging.error(f"Exception during hydration: {str(e)}")
        return False

def extract_tweets_from_jsonl(jsonl_file, csv_output):
    """
    Extracts relevant fields from twarc2 JSONL output.
     Twarc2 returns tweets nested under 'data' in each line.
    """
    import pandas as pd
    
    records = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if 'data' in obj:
                        for tweet in obj['data']:
                            records.append({
                                'tweet_id': tweet.get('id'),
                                'created_at': tweet.get('created_at'),
                                'text': tweet.get('text', ''),
                                'lang': tweet.get('lang', ''),
                                'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                                'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                                'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                                'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                                'author_id': tweet.get('author_id')
                            })
                except json.JSONDecodeError:
                    logging.warning("Skipped invalid JSON line.")
                    
        df = pd.DataFrame(records)
        df.to_csv(csv_output, index=False)
        logging.info(f"Extracted {len(df)} tweets to {csv_output}")
        return df
    
    except Exception as e:
        logging.error(f"Error parsing JSONL {jsonl_file}: {str(e)}")
        return pd.DataFrame()

def run_hydration_pipeline():
    """
    Orchestrates hydration of all .txt files in RAW_DIR.
    """
    os.makedirs(HYDRATED_DIR, exist_ok=True)
    raw_files = glob(os.path.join(RAW_DIR, "*.txt"))
    
    logging.info(f"Found {len(raw_files)} raw ID files to process.")
    
    for raw_file in raw_files:
        basename = os.path.basename(raw_file).replace('.txt', '')
        jsonl_out = os.path.join(HYDRATED_DIR, f"{basename}.jsonl")
        csv_out = os.path.join(HYDRATED_DIR, f"{basename}.csv")
        
        if os.path.exists(csv_out):
            logging.info(f"Skipping {basename}, already processed into CSV.")
            continue
            
        if not os.path.exists(jsonl_out):
            success = hydrate_ids(raw_file, jsonl_out)
            if not success:
                continue
                
        # If jsonl exists (or was just created), extract it
        if os.path.exists(jsonl_out):
             extract_tweets_from_jsonl(jsonl_out, csv_out)
             
if __name__ == '__main__':
    run_hydration_pipeline()
