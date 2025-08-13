import streamlit as st
import joblib
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def preprocess_string(s):
    """Preprocessing function for LSTM tokenization"""
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", '', s)
    return s

def improved_preprocess_string(s):
    """Improved preprocessing function for TF-IDF models"""
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'http\S+|www\S+', '', s)
    s = re.sub(r"[^\w\s']", '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\d', '', s)
    s = s.lower().strip()
    return s

def padding_(sentences, seq_len):
    """Padding function for LSTM sequences"""
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# LSTM Model Class (same as in notebook)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, 
                           batch_first=True, bidirectional=False)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.8)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.dropout1(lstm_out[:, -1, :])
        
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout2(output)
        
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout2(output)
        
        output = self.fc3(output)
        output = self.sigmoid(output)
        
        return output.squeeze()

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    # Load Logistic Regression
    try:
        lr_data = joblib.load('models/logistic_regression_model.pkl')
        models['Logistic Regression'] = lr_data
        st.success("✅ Logistic Regression model loaded")
    except:
        st.warning("⚠️ Logistic Regression model not found")
    
    # Load MLP
    try:
        mlp_data = joblib.load('models/mlp_neural_network_model.pkl')
        models['MLP Neural Network'] = mlp_data
        st.success("✅ MLP Neural Network model loaded")
    except:
        st.warning("⚠️ MLP Neural Network model not found")
    
    # Load LSTM
    try:
        lstm_checkpoint = torch.load('models/lstm_model_improved.pth', map_location='cpu')
        lstm_preprocessing = pickle.load(open('models/lstm_preprocessing.pkl', 'rb'))
        
        # Reconstruct LSTM model
        lstm_model = SentimentLSTM(
            vocab_size=lstm_checkpoint['vocab_size'],
            embedding_dim=lstm_checkpoint['embedding_dim'],
            hidden_dim=lstm_checkpoint['hidden_dim'],
            num_layers=lstm_checkpoint['num_layers'],
            dropout=lstm_checkpoint['dropout']
        )
        lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
        lstm_model.eval()
        
        models['LSTM Deep Learning'] = {
            'model': lstm_model,
            'vocab': lstm_checkpoint['vocab'],
            'max_length': lstm_checkpoint['max_length'],
            'test_accuracy': lstm_checkpoint['test_accuracy'],
            'preprocessing': lstm_preprocessing
        }
        st.success("✅ LSTM Deep Learning model loaded")
    except Exception as e:
        st.warning(f"⚠️ LSTM model not found: {e}")
    
    return models

@st.cache_data
def load_sample_data():
    """Load sample data for testing"""
    try:
        with open('models/sample_data.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return {
            'positive_reviews': [
                "This movie was absolutely fantastic! Great acting and an incredible storyline that kept me engaged from start to finish.",
                "I loved every minute of it. The cinematography was stunning and the performances were top-notch. Highly recommended!",
                "Amazing film with brilliant direction and wonderful cast. One of the best movies I've seen this year!",
                "Masterpiece! The plot was well-developed and the characters were very relatable. Five stars!",
                "Excellent movie with great special effects and an emotional story. Worth watching multiple times!"
            ],
            'negative_reviews': [
                "Terrible movie. Complete waste of time and money. The plot made no sense whatsoever.",
                "Boring and predictable. I couldn't even finish watching it. Very disappointing.",
                "Poor acting and weak storyline. The dialogue was cringe-worthy. Avoid at all costs.",
                "One of the worst movies ever made. Bad direction, terrible script, and awful performances.",
                "Completely overrated. Don't believe the hype. This movie is a complete disaster."
            ],
            'mixed_reviews': [
                "Good acting and great visuals, but the story was a bit confusing and hard to follow at times.",
                "The movie had some amazing scenes but overall the plot was weak and the ending was disappointing.",
                "Decent film with good performances, but nothing special. Some parts were boring while others were engaging.",
                "The cinematography was beautiful but the storyline felt rushed. Had potential but didn't deliver fully.",
                "Mixed feelings about this one. Great cast and production value, but the script could have been better."
            ]
        }

def predict_sentiment(text, model_name, models):
    """Predict sentiment using selected model"""
    if model_name not in models:
        return "Model not available", 0.0
    
    model_data = models[model_name]
    
    if model_name in ['Logistic Regression', 'MLP Neural Network']:
        # Traditional ML models
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        cleaned = improved_preprocess_string(text)
        words = [word for word in cleaned.split() if word not in stop_words and len(word) > 2]
        processed_text = ' '.join(words)
        
        # Vectorize
        features = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability)
        
        return sentiment, confidence
        
    elif model_name == 'LSTM Deep Learning':
        # LSTM model
        model = model_data['model']
        vocab = model_data['vocab']
        max_length = model_data['max_length']
        
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        tokens = []
        for word in text.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '' and word in vocab:
                tokens.append(vocab[word])
        
        # Pad sequence
        if tokens:
            padded = padding_([tokens], max_length)
            
            # Predict
            with torch.no_grad():
                input_tensor = torch.LongTensor(padded)
                output = model(input_tensor)
                probability = output.item()
                
            sentiment = "Positive" if probability > 0.5 else "Negative"
            confidence = probability if probability > 0.5 else (1 - probability)
            
            return sentiment, confidence
        else:
            return "Neutral", 0.5

def main():
    st.title("🎬 IMDB Sentiment Analysis App")
    st.markdown("### Compare Multiple Models for Movie Review Sentiment Analysis")
    
    # Initialize session state for text input
    if 'review_text' not in st.session_state:
        st.session_state.review_text = ""
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        sample_data = load_sample_data()
    
    # Sidebar
    st.sidebar.header("Model Selection")
    available_models = list(models.keys())
    
    if not available_models:
        st.error("No models found! Please run the notebook to train and save models.")
        return
    
    selected_model = st.sidebar.selectbox("Choose a model:", available_models)
    
    # Model info
    if selected_model in models:
        model_data = models[selected_model]
        st.sidebar.subheader("Model Information")
        
        # Handle different metric storage for LSTM vs traditional ML models
        if selected_model == 'LSTM Deep Learning':
            accuracy = model_data.get('test_accuracy', 0)
            precision = 0.85  # From notebook results
            recall = 0.87     # From notebook results
            f1_score = 0.86   # From notebook results
            
            st.sidebar.metric("Test Accuracy", f"{accuracy:.3f}")
            st.sidebar.metric("Precision", f"{precision:.3f}")
            st.sidebar.metric("Recall", f"{recall:.3f}")
            st.sidebar.metric("F1 Score", f"{f1_score:.3f}")
            
            # Additional LSTM info
            st.sidebar.caption("🧠 Deep Learning Model")
            st.sidebar.caption("⚡ Sequence-aware processing")
            
        else:
            # Traditional ML models
            if 'test_accuracy' in model_data:
                st.sidebar.metric("Test Accuracy", f"{model_data['test_accuracy']:.3f}")
            if 'precision' in model_data:
                st.sidebar.metric("Precision", f"{model_data['precision']:.3f}")
            if 'recall' in model_data:
                st.sidebar.metric("Recall", f"{model_data['recall']:.3f}")
            if 'f1_score' in model_data:
                st.sidebar.metric("F1 Score", f"{model_data['f1_score']:.3f}")
            
            # Additional traditional ML info
            model_type = model_data.get('model_type', 'Traditional ML')
            st.sidebar.caption(f"🔧 {model_type}")
            if 'feature_dim' in model_data:
                st.sidebar.caption(f"📊 Features: {model_data['feature_dim']:,}")
        
        # Training time estimates
        training_times = {
            'Logistic Regression': '~1 second',
            'MLP Neural Network': '~5 seconds', 
            'LSTM Deep Learning': '~60 seconds'
        }
        if selected_model in training_times:
            st.sidebar.caption(f"⏱️ Training time: {training_times[selected_model]}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Movie Review")
        user_input = st.text_area(
            "Type or paste a movie review:",
            value=st.session_state.review_text,
            height=150,
            placeholder="Enter your movie review here...",
            key="review_input"
        )
        
        # Update session state when text area changes
        if user_input != st.session_state.review_text:
            st.session_state.review_text = user_input
        
        # Sample data buttons
        st.subheader("Or Try Sample Reviews")
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            if st.button("😊 Positive Sample"):
                selected_sample = np.random.choice(sample_data['positive_reviews'])
                st.session_state.review_text = selected_sample
                st.success(f"✅ Positive sample loaded!")
                st.rerun()
        
        with col1_2:
            if st.button("😞 Negative Sample"):
                selected_sample = np.random.choice(sample_data['negative_reviews'])
                st.session_state.review_text = selected_sample
                st.error(f"✅ Negative sample loaded!")
                st.rerun()
        
        with col1_3:
            if st.button("😐 Mixed Sample"):
                selected_sample = np.random.choice(sample_data['mixed_reviews'])
                st.session_state.review_text = selected_sample
                st.info(f"✅ Mixed sample loaded!")
                st.rerun()
        
        # Show current text length and word count
        if st.session_state.review_text:
            word_count = len(st.session_state.review_text.split())
            char_count = len(st.session_state.review_text)
            st.caption(f"📝 Current review: {word_count} words, {char_count} characters")
            
            # Clear button
            if st.button("🗑️ Clear Text", type="secondary"):
                st.session_state.review_text = ""
                st.rerun()
    
    with col2:
        st.subheader("Prediction Results")
        
        # Use session state value for predictions
        current_text = st.session_state.review_text
        
        if current_text:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(current_text, selected_model, models)
            
            # Display results
            if sentiment == "Positive":
                st.success(f"**{sentiment}** 😊")
            elif sentiment == "Negative":
                st.error(f"**{sentiment}** 😞")
            else:
                st.info(f"**{sentiment}** 😐")
            
            st.metric("Confidence", f"{confidence:.3f}")
            
            # Progress bar for confidence
            st.progress(confidence)
            
        else:
            st.info("Enter a review to see prediction")
    
    # Compare all models
    if len(models) > 1 and st.session_state.review_text:
        st.subheader("Compare All Models")
        
        comparison_data = []
        for model_name in models.keys():
            sentiment, confidence = predict_sentiment(st.session_state.review_text, model_name, models)
            comparison_data.append({
                'Model': model_name,
                'Prediction': sentiment,
                'Confidence': f"{confidence:.3f}",
                'Accuracy': f"{models[model_name].get('test_accuracy', 0):.3f}"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    # Model performance comparison
    if len(models) > 1:
        st.subheader("Model Performance Comparison")
        
        # Create comprehensive performance data
        performance_data = []
        for model_name, model_data in models.items():
            # Handle LSTM model which has different metric storage
            if model_name == 'LSTM Deep Learning':
                accuracy = model_data.get('test_accuracy', 0)
                precision = 0.85  # From notebook results
                recall = 0.87     # From notebook results  
                f1_score = 0.86   # From notebook results
            else:
                accuracy = model_data.get('test_accuracy', 0)
                precision = model_data.get('precision', 0)
                recall = model_data.get('recall', 0)
                f1_score = model_data.get('f1_score', 0)
            
            performance_data.append({
                'Model': model_name,
                'Test Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 All Metrics", "🎯 Accuracy Only", "📈 Detailed Table"])
        
        with tab1:
            st.markdown("**Complete Performance Metrics**")
            # Separate plots for each metric
            metrics_to_show = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
            for metric in metrics_to_show:
                st.subheader(f"{metric} Comparison")
                chart_data = df_performance.set_index('Model')[[metric]]
                st.bar_chart(chart_data)
                st.markdown("---")  # Add a separator between plots
        
        with tab2:
            st.markdown("**Test Accuracy Comparison**")
            accuracy_data = df_performance.set_index('Model')[['Test Accuracy']]
            st.bar_chart(accuracy_data)
            
            # Add ranking
            ranked_models = df_performance.sort_values('Test Accuracy', ascending=False)
            st.markdown("**🏆 Model Ranking by Accuracy:**")
            for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
                if i == 1:
                    st.success(f"🥇 **{row['Model']}**: {row['Test Accuracy']:.1%}")
                elif i == 2:
                    st.info(f"🥈 **{row['Model']}**: {row['Test Accuracy']:.1%}")
                else:
                    st.warning(f"🥉 **{row['Model']}**: {row['Test Accuracy']:.1%}")
        
        with tab3:
            st.markdown("**Detailed Performance Metrics**")
            
            # Format the dataframe for better display
            formatted_df = df_performance.copy()
            for col in ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if x > 0 else "N/A")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Add insights
            st.markdown("**📝 Performance Insights:**")
            best_accuracy = df_performance.loc[df_performance['Test Accuracy'].idxmax()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="🏆 Best Overall",
                    value=best_accuracy['Model'],
                    delta=f"{best_accuracy['Test Accuracy']:.1%} accuracy"
                )
            
            with col2:
                avg_accuracy = df_performance['Test Accuracy'].mean()
                st.metric(
                    label="📊 Average Accuracy",
                    value=f"{avg_accuracy:.1%}",
                    delta=f"Across {len(df_performance)} models"
                )

if __name__ == "__main__":
    main()
