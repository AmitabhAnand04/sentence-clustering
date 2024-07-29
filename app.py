from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
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

# # Compute cosine distance
# cosine_dist = cosine_distances(sentence_embeddings)

# # Ensure the distance matrix is of type float64
# cosine_dist = cosine_dist.astype(np.float64)

# Apply K-Means clustering
num_clusters = 8  # Adjust based on expected number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(sentence_embeddings)

# Analyze clusters
clustered_sentences = {}
for sentence, cluster in zip(sentences, clusters):
    if cluster not in clustered_sentences:
        clustered_sentences[cluster] = []
    clustered_sentences[cluster].append(sentence)

# for cluster, sentences in clustered_sentences.items():
#     print(f"Cluster {cluster}:")
#     for sentence in sentences:
#         print(f"  - {sentence}")
# Step 8: Write clusters to a file
with open('clustered_sentences.txt', 'w') as file:
    for cluster, sentences in clustered_sentences.items():
        print(f"Cluster {cluster}:")
        file.write(f"Cluster {cluster}:\n")
        for sentence in sentences:
            print(f"  - {sentence}")
            file.write(f"  - {sentence}\n")
        file.write("\n")

# Step 9: Print a summary of clusters
print("Summary of clusters:")
for cluster, sentences in clustered_sentences.items():
    print(f"Cluster {cluster}: {len(sentences)} sentences")

print("Detailed cluster information has been written to 'clustered_sentences.txt'")
