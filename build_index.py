from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your text
with open("training_data.txt", "r") as file:
    text = file.read()

# Split into chunks (simple version)
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Build a search index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save everything
faiss.write_index(index, "data_index.faiss")
np.save("chunks.npy", chunks)

print("Data processed and saved!")