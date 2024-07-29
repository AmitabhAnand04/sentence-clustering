import dask.dataframe as dd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

# Load data into a Dask DataFrame
df = dd.read_csv('output.csv')  # Replace with your data file

# Convert sentences to vectors using Dask
model = SentenceTransformer('all-MiniLM-L6-v2')
def process_batch(batch):
    return model.encode(batch['sentences'].tolist())

embeddings = df.map_partitions(process_batch, meta=('x', 'f8')).compute()

# Apply Mini-Batch K-Means clustering
mini_batch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=0)
clusters = mini_batch_kmeans.fit_predict(embeddings)

# Analyze clusters (handle this part separately for efficiency)
