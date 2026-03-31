**From Fear to Forecast: ML Solutions for Mass Psychosis in Crisis Scenarios**

📌 Overview - 
Mass hysteria, or Mass Psychogenic Illness (MPI), is the rapid spread of anxiety, fear, or irrational behavior among groups without identifiable organic causes. The COVID-19 pandemic demonstrated how public health crises, combined with high-velocity digital communication, can trigger collective emotional responses and panic. 

This project utilizes Machine Learning (ML), Natural Language Processing (NLP), and time-series forecasting to identify, analyze, and predict early warning signs of mass hysteria using real-time social media data. By understanding the temporal dynamics of public sentiment, this framework provides actionable insights for crisis management, public health communication, and misinformation mitigation.

✨ Key Findings  
Our analysis maps public sentiment during crises into three distinct phases:
* **Phase 1 (Escalation & Positive Amplification):** Early surge of negative sentiment driven by fear and initial panic.
* **Phase 2 (Saturation & Sentiment Decay):** A steep decline into intense negativity, likely induced by misinformation surges or escalating crisis conditions.
* **Phase 3 (Stabilization & Potential Recovery):** Stabilization as positive and negative sentiment oscillations reduce over time.

## 📊 Dataset 
This project relies on Twitter data to capture public sentiment and reactions to the pandemic in real-time. 

**Data Source:** The primary dataset used is the CML-COVID dataset, which comprises approximately 20 million tweets from over 600,000 users, collected during March and July 2020 using pandemic-relevant trigger words. 

> **Citation:** Dashtian, H., & Murthy, D. (2021). CML-COVID: A Large-Scale COVID-19 Twitter Dataset with Latent Topics, Sentiment and Location Information. ArXiv.(https://arxiv.org/abs/2101.12202)

## ⚙️ Methodology Pipeline  
The methodological framework is structured into the following interdependent components:
1.  **Data Preprocessing:** Segmenting text via tokenization, applying lemmatization to normalize morphological variations, converting to lowercase, and removing stop-words, URLs, mentions, emojis, and special characters to reduce noise.
2.  **Feature Engineering:** Transforming textual data into numerical representations using TF-IDF and word embeddings to capture context. Network diffusion attributes (likes, retweets) and temporal data (timestamps) are also integrated to contextualize emotional contagion.
3.  **Sentiment Analysis & Forecasting:** Utilizing VADER for emotional intensity prediction and Long Short-Term Memory (LSTM) networks to analyze time-series sentiment fluctuations.
4.  [**Classification:** Employing multiple ML architectures to classify risk levels and indicate the likelihood of mass hysteria events.

## 🚀 Model Performance 
We evaluated traditional statistical learning models against deep learning and transformer-based architectures. Transformer-based models, specifically BERT, demonstrated superior capability in capturing contextual linguistic relationships and sentiment polarity.

## 🛠 Tech Stack 
* **Environment:** Python 3.10, Google Colab (GPU: NVIDIA T4, 16GB RAM) 
* **Data Processing & Modeling:** Scipy, RPAPE, VADER 
* **Visualization:** Matplotlib (for detailed graphical representations of sentiment shifts) 

## 👥 Authors 
* **Raahil Sheikh** (Chief Technology Officer, Beyond Space Technologies) 
* **Lavleen** (Department of Computer Science and Engineering, Chandigarh University) 
* **Disha Singh** (Department of Computer Science and Engineering, Chandigarh University) 
* **Pragyan Priyadarshini Rout** (Department of Computer Science and Engineering, Chandigarh University) 
* **Shreya Das** (Department of Computer Science and Engineering, Chandigarh University) 
