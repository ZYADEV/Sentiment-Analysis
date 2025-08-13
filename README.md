# IMDB Sentiment Analysis

This project is a Streamlit web application for sentiment analysis of movie reviews. It utilizes various machine learning and deep learning models to predict whether a review is positive or negative.

## Features

*   **Sentiment Analysis:** Classifies movie reviews as either positive or negative.
*   **Multiple Models:** Implements and compares the following models:
    *   Logistic Regression
    *   Multilayer Perceptron (MLP)
    *   Long Short-Term Memory (LSTM)
*   **Model Comparison:** Provides a detailed comparison of the performance metrics of each model.
*   **Interactive Interface:** Allows users to input their own reviews or use sample reviews for testing.
*   **Detailed Metrics:** Displays accuracy, precision, recall, and F1-score for each model.

## Dataset

The project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset contains 50,000 movie reviews, evenly split into 25,000 for training and 25,000 for testing.

## Models

The following models are implemented in this project:

*   **Logistic Regression:** A linear model for binary classification.
*   **Multilayer Perceptron (MLP):** A feedforward artificial neural network.
*   **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) well-suited for sequence data like text.

## How to Run

1.  **Install the dependencies:**
    ```bash
    pip install -r streamlit_requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run sentiment_analysis_streamlit_app.py
    ```

## Demo

[Demo Video](https://github.com/ZYADEV/Sentiment-Analysis/blob/main/Demo%20Video/IMDB%20Sentiment%20Analysis.mp4)
