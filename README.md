        Twitter Sentiment Analysis with LSTM and Class Weights
        
Project Overview

This project implements a sentiment analysis model for Twitter data using a Bidirectional Long Short-Term Memory (BiLSTM) neural network with TensorFlow/Keras. The goal is to classify tweets into 'negative', 'neutral', or 'positive' sentiments. A key aspect of this implementation is the use of class weights during training to address potential class imbalance and improve the model's ability to predict minority classes (e.g., 'negative' sentiment).

The project follows a standard machine learning workflow, from data loading and preprocessing to model training, evaluation, and saving for future deployment (e.g., in a Streamlit application).

Workflow

Dataset Overview and Basic EDA: Initial loading and inspection of Twitter_Data.csv to understand its structure, sentiment distribution, and text characteristics.

1.Text Cleaning (NLP Preprocessing) : Development of a clean_text() function to preprocess tweets by removing URLs, mentions, lowercasing, and handling punctuation. This ensures consistent data format for model training and inference.

2.Train-Test Split: Splitting the cleaned data into training and testing sets, using a stratified approach to maintain class distribution across splits.

3.Tokenization and Sequence Padding: Converting text data into numerical sequences using a Keras Tokenizer, followed by padding/truncating sequences to a uniform length suitable for the LSTM model.

4.Building a BiLSTM Model: Constructing a deep learning model in TensorFlow/Keras featuring an Embedding layer, SpatialDropout1D, a Bidirectional LSTM layer, and Dense layers with Dropout for classification.

5.Training with EarlyStopping and Class Weights: Training the model using the Adam optimizer and sparse_categorical_crossentropy loss. Crucially, compute_class_weight is used to address class imbalance, and EarlyStopping is employed to prevent overfitting.

6.Model Evaluation: Comprehensive evaluation of the trained model on the test set using a classification report (precision, recall, F1-score) and confusion matrix to assess performance across all sentiment classes.

7.Saving Model and Tokenizer: Persisting the trained Keras model (.h5 format) and the fitted Tokenizer (.joblib format) for seamless integration into a deployment application.
Technologies Used
Python

8.TensorFlow/Keras: For building and training the deep learning model.
Pandas: For data manipulation and analysis.
scikit-learn: For data splitting and evaluation metrics.
Joblib: For saving the tokenizer.
re: For regular expression-based text cleaning.

Install dependencies:
pip install -r requirements.txt
(Make sure requirements.txt includes tensorflow, pandas, scikit-learn, joblib)

Download the dataset: Place Twitter_Data.csv in the project root or update the DATA_PATH variable in the notebook.

Run the Jupyter Notebook/Colab: Open and run sentiment_analysis_lstm.ipynb (or similar) cell by cell to execute the full workflow.

Deployment (Optional): The saved sentiment_model.h5 and tokenizer.joblib can be used to build a Streamlit application for real-time sentiment prediction. Refer to app.py (if provided separately) for deployment details.

Results

The model achieves good accuracy and F1-scores across all three sentiment classes, demonstrating that the use of class weights effectively mitigated the imbalance issues observed in earlier iterations, leading to improved recall for the 'negative' class.
