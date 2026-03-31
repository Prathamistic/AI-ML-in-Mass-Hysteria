import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_changepoints(data, pen=10, model="l2"):
    """
    Detects changepoints in the sequence using the PELT algorithm.
    Args:
        data: shape (n_samples, n_features)
        pen: Penalty for the number of changepoints.
        model: Ruptures model type (e.g., "l2", "rbf").
    """
    logging.info(f"Detecting changepoints with PELT (model={model}, penalty={pen})...")
    # PELT search
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=pen)
    
    logging.info(f"Detected {len(result)-1} changepoints at indices: {result[:-1]}")
    return result

def cluster_hidden_states(hidden_states, n_clusters=3):
    """
    Clusters the hidden states using KMeans to identify distinct phases.
    """
    logging.info(f"Clustering hidden states into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(hidden_states)
    
    score = silhouette_score(hidden_states, cluster_labels)
    logging.info(f"Silhouette Score: {score:.4f}")
    
    return cluster_labels, score

def map_phases(df_daily, changepoints, cluster_labels):
    """
    Maps detected changepoints and clusters back to the daily dataframe.
    Note: changepoints from PELT include the last index.
    Hidden states and cluster labels are associated with the sliding windows.
    We need to align them back to the original dates.
    """
    # Assuming the LSTM window size was fixed, the first window ends at window_size-1
    # We'll align the labels to the end of the windows.
    # If df_daily has N rows and window_size is W, we have N-W+1 windows.
    # The first label corresponds to row index W-1.
    
    window_size = len(df_daily) - len(cluster_labels) + 1
    
    df_result = df_daily.copy()
    df_result['cluster'] = np.nan
    df_result.iloc[window_size-1:, df_result.columns.get_loc('cluster')] = cluster_labels
    
    # Track changepoints
    df_result['is_changepoint'] = False
    # Indices are relative to the input data of PELT (which was hidden_states)
    # So index 'i' in changepoints corresponds to row window_size-1 + i in df_result
    for cp in changepoints[:-1]: # exclude the last one which is n_samples
        idx = window_size - 1 + cp
        if idx < len(df_result):
            df_result.iloc[idx, df_result.columns.get_loc('is_changepoint')] = True
            
    return df_result

def get_phase_names(df_result):
    """
    Assigns human-readable names to phases based on chronological order of clusters.
    """
    phases = []
    current_phase = 0
    phase_starts = df_result[df_result['is_changepoint']].index.tolist()
    
    # This is a simplified mapping logic
    # In a real study, labels would be assigned after inspecting cluster characteristics
    df_result['phase_name'] = df_result['cluster'].apply(lambda x: f"Phase {int(x)}" if not np.isnan(x) else "Initial")
    
    return df_result
