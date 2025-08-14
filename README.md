Jailbreak Hackathon Project – Prompt Classification

This project was developed as part of the MIT Bangalore hackathon.
The main goal was to build a system that can classify prompts using embeddings generated from a pre-trained language model and then train a classifier to categorize them accurately.

My Contribution

I worked specifically on the scripts and setup for generating embeddings, training the classifier, and preparing files for handover.
On Day 1, I wrote and tested the embedding_script.py file. This script loads the dataset of prompts from a CSV file, uses a Sentence Transformer model to generate embeddings, and saves those embeddings for later use.

On Day 2, I created the train_classifier.py file. This script loads the embeddings, splits the data into training and testing sets, trains a classification model, and evaluates its performance.

On Day 3, I prepared the files for handover, which included:

The trained model file (classifier_model.pkl)

The dataset file (prompts.csv)

Both Python scripts (embedding_script.py and train_classifier.py)

These files were shared with my teammate  to integrate into the final system.

Tools and Libraries Used

Python

Pandas – for loading and managing data

NumPy – for handling arrays and numerical operations

Sentence Transformers – for generating high-quality embeddings from text

Scikit-learn – for training and evaluating the classifier model













