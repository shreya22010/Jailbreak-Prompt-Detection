# embedding_script.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Step 1: Load your dataset
data_path = "../data/prompts.csv"
df = pd.read_csv(data_path)

# Step 2: Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Generate embeddings from prompts
embeddings = model.encode(df['prompt'].tolist())

# Step 4: Save embeddings and labels for training later
os.makedirs("../data", exist_ok=True)
np.save("../data/embeddings.npy", embeddings)
np.save("../data/labels.npy", df['label'].values)

print("✅ Embeddings and labels saved in data/ folder!")
