# Jailbreak Prompt Classification

A machine learning system that classifies prompts using embeddings from a pre-trained language model.

## What it does
- Loads prompt datasets and generates embeddings using Sentence Transformers
- Trains a classifier to categorize prompts accurately
- Evaluates model performance on test data

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Sentence Transformers, Scikit-learn

## Project Structure
- `embedding_script.py` - Loads dataset, generates and saves embeddings
- `train_classifier.py` - Trains and evaluates the classification model
- `classifier_model.pkl` - Trained model file
- `prompts.csv` — Dataset file
