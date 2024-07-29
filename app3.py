from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Prepare your sentences
sentences = [
    "How do I reset my password?",
    "How can I change my password?",
    "What is the process to recover my account?",
    "How to contact customer support?",
    "Where can I find customer service contact?",
    "What is the customer support number?",
]

# Convert sentences to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)

# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='average')
clusters = agglo.fit_predict(sentence_embeddings)

# Analyze clusters
clustered_sentences = {}
for sentence, cluster in zip(sentences, clusters):
    if cluster not in clustered_sentences:
        clustered_sentences[cluster] = []
    clustered_sentences[cluster].append(sentence)

for cluster, sentences in clustered_sentences.items():
    print(f"Cluster {cluster}:")
    for sentence in sentences:
        print(f"  - {sentence}")
