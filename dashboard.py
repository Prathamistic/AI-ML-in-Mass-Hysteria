import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_metrics(metrics_dict, output_file):
    """
    Saves metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"Metrics saved to {output_file}")

def create_research_dashboard(df_result, hidden_states, reconstruction_loss, baseline_loss, output_file):
    """
    Generates a 6-panel publication-quality research dashboard.
    """
    sns.set_palette("viridis")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle("Twitter Crisis-Signal Analysis Dashboard (2020)", fontsize=22)
    
    df_plot = df_result.reset_index()
    
    # Panel 1: Sentiment Trend (mean_sentiment)
    sns.lineplot(ax=axes[0,0], x='date', y='mean_sentiment', data=df_plot, color='blue')
    axes[0,0].set_title("Daily Mean Sentiment Trend", fontsize=16)
    axes[0,0].set_ylabel("VADER Compound Score")
    
    # Panel 2: Rumor Velocity & Crisis Keyword Density
    sns.lineplot(ax=axes[0,1], x='date', y='rumor_velocity', data=df_plot, label='Rumor Velocity', color='red')
    sns.lineplot(ax=axes[0,1], x='date', y='crisis_kw_density', data=df_plot, label='Crisis KW Density', color='orange')
    axes[0,1].set_title("Rumor velocity vs Crisis Keyword Density", fontsize=16)
    axes[0,1].set_ylabel("Density (Relative to Volume)")
    axes[0,1].legend()

    # Panel 3: Distress Heatmap (weekly)
    df_distress = df_plot.copy()
    df_distress['week'] = pd.to_datetime(df_distress['date']).dt.isocalendar().week
    distress_pivot = df_distress.pivot_table(index='week', values='mean_distress', aggfunc='mean')
    sns.heatmap(distress_pivot.T, ax=axes[1,0], cmap="YlOrRd", annot=True)
    axes[1,0].set_title("Weekly Distress Intensity Heatmap", fontsize=16)

    # Panel 4: Hidden-State Projection (PCA)
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(hidden_states)
    n_windows = len(h_pca)
    cluster_plot = df_plot.iloc[-n_windows:].copy() # Aligning back to clusters
    cluster_plot['PCA1'] = h_pca[:, 0]
    cluster_plot['PCA2'] = h_pca[:, 1]
    
    sns.scatterplot(ax=axes[1,1], x='PCA1', y='PCA2', hue='cluster', data=cluster_plot, palette='viridis', alpha=0.7)
    axes[1,1].set_title("Hidden-State Projection (PCA)", fontsize=16)

    # Panel 5: Phase Summary Transition Table (as a horizontal bar for segments)
    # Mapping phase boundaries visually
    current_phase = cluster_plot['cluster'].iloc[0]
    for cp in df_plot[df_plot['is_changepoint']].date:
        axes[2,0].axvline(cp, color='black', linestyle='--')
    
    sns.lineplot(ax=axes[2,0], x='date', y='cluster', data=df_plot, marker='o', drawstyle='steps-post')
    axes[2,0].set_title("Detected Phase Transitions over Time", fontsize=16)
    axes[2,0].set_yticks(sorted(df_plot['cluster'].dropna().unique()))
    axes[2,0].set_ylabel("Cluster ID")

    # Panel 6: Reconstruction Performance (Comparison)
    perf_data = pd.DataFrame({
        'Model': ['Persistence Baseline', 'LSTM Autoencoder'],
        'MSE Loss': [baseline_loss, reconstruction_loss]
    })
    sns.barplot(ax=axes[2,1], x='Model', y='MSE Loss', data=perf_data, palette='coolwarm')
    axes[2,1].set_title("Reconstruction MSE: LSTM AE vs Baseline", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    logging.info(f"Dashboard saved to {output_file}")
    plt.close()

def compute_baseline_mse(X_scaled):
    """
    Computes a persistence baseline (predicting X_{t} = X_{t-1}).
    For an autoencoder, a naive baseline is predicting the mean or X_t itself.
    Usually, for reconstruction, an identity is the trivial baseline but here 
    the AE compresses the data. A reasonable comparison is a 'Persistence' 
    or a mean-field reconstruction.
    """
    # Simply using mean of features as a basic baseline
    mean_val = np.mean(X_scaled, axis=0)
    baseline_recon = np.tile(mean_val, (X_scaled.shape[0], 1, 1))
    mse = np.mean((X_scaled - baseline_recon)**2)
    return mse
