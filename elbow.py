from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Step 1: Read the CSV file
csv_file_path = 'output.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Step 2: Extract sentences, remove any empty rows
sentences = df['convPrompt'].dropna().tolist()

# Convert sentences to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)

# Find optimal number of clusters using the Elbow Method
inertia = []
for num_clusters in range(1, 10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_embeddings)
    inertia.append(kmeans.inertia_)

# Plot the results
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
