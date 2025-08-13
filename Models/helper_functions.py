
import numpy as np
import re
from nltk.corpus import stopwords

def preprocess_string(s):
    """Preprocessing function for LSTM tokenization"""
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s

def improved_preprocess_string(s):
    """Improved preprocessing function for TF-IDF models"""
    import re
    
    # Remove HTML tags
    s = re.sub(r'<[^>]+>', '', s)
    
    # Remove URLs
    s = re.sub(r'http\S+|www\S+', '', s)
    
    # Remove non-word characters but keep apostrophes for contractions
    s = re.sub(r"[^\w\s']", '', s)
    
    # Replace multiple whitespaces with single space
    s = re.sub(r'\s+', ' ', s)
    
    # Remove digits
    s = re.sub(r'\d', '', s)
    
    # Convert to lowercase and strip
    s = s.lower().strip()
    
    return s

def padding_(sentences, seq_len):
    """Padding function for LSTM sequences"""
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features
