import yaml
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np

# Load CodeBERT - disable authentication
import os
# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Disable HF authentication
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

model = AutoModel.from_pretrained("microsoft/codebert-base", trust_remote_code=True, use_safetensors=False, token=False)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=True, token=False)
model.eval()

def embed_text(text: str, max_tokens: int = 512) -> np.ndarray:
    """Embed long text by splitting into chunks and averaging CodeBERT embeddings."""
    embeddings = []
    
    # Split the text into chunks of approximately max_tokens length
    # We'll use a simple character-based approach for chunking
    text_length = len(text)
    chunk_size = max_tokens * 4  # Rough estimate of characters per token
    
    for i in range(0, text_length, chunk_size):
        chunk_text = text[i:i + chunk_size]
        
        # Encode the chunk directly as a string
        chunk_ids = tokenizer.encode(chunk_text, return_tensors="pt", 
                                    max_length=max_tokens,
                                    truncation=True,
                                    add_special_tokens=True)
        
        with torch.no_grad():
            outputs = model(chunk_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings.append(cls_embedding[0].cpu().numpy())
    
    # If no embeddings were created (e.g., empty text), return a zero vector
    if not embeddings:
        return np.zeros(768)  # CodeBERT embeddings are 768-dimensional
    
    # Return raw mean embedding (no normalization); we will center & normalize later
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding

def yaml_to_text(pr_data: dict) -> str:
    """Convert PR YAML data to a detailed text representation with anchors, ignoring operation type."""
    lines = [f"PR ID: {pr_data['pr_id']}", ""]
    
    # Add a more detailed representation with anchors, but ignore operation type
    for file in pr_data["file_changes"]:
        file_path = file['path'] + file['filename']
        lines.append(f"FILE: {file_path}")
        
        for cls in file.get("class_changes", []):
            class_anchor = f"{file_path}::{cls['class_name']}"
            lines.append(f"CLASS: {class_anchor}")
            
            for fn in cls.get("function_changes", []):
                func_anchor = f"{class_anchor}::{fn['function_name']}"
                lines.append(f"FUNCTION: {func_anchor}")
                # Add the function signature to provide more context
                lines.append(f"SIGNATURE: {fn['function_name']}")
    
    # Join with double newlines to give more weight to the separation
    return "\n\n".join(lines)

def extract_anchors(pr_data: dict) -> dict:
    """Extract anchors at file, class, and function levels for hierarchical matching."""
    anchors = {
        'file': set(),
        'class': set(),
        'function': set()
    }
    
    for file in pr_data.get("file_changes", []):
        file_path = file["path"] + file["filename"]
        anchors['file'].add(file_path)
        
        for cls in file.get("class_changes", []):
            class_anchor = f"{file_path}::{cls['class_name']}"
            anchors['class'].add(class_anchor)
            
            for fn in cls.get("function_changes", []):
                func_anchor = f"{class_anchor}::{fn['function_name']}"
                anchors['function'].add(func_anchor)
    
    return anchors

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def hierarchical_similarity(a: dict, b: dict) -> float:
    """Calculate weighted similarity across file, class, and function levels."""
    if not a or not b:
        return 0.0
    
    # Calculate Jaccard similarity at each level
    file_sim = jaccard(a.get('file', set()), b.get('file', set()))
    class_sim = jaccard(a.get('class', set()), b.get('class', set()))
    func_sim = jaccard(a.get('function', set()), b.get('function', set()))
    
    # Prioritize file and class matches more heavily
    # Weight the similarities: files (40%), classes (40%), functions (20%)
    return 0.4 * file_sim + 0.4 * class_sim + 0.2 * func_sim

def build_faiss_index(embeddings: List[np.ndarray]) -> faiss.Index:
    dim = embeddings[0].shape[0]
    # Use IndexFlatIP for inner product, which is equivalent to cosine similarity for normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(np.stack(embeddings).astype('float32'))
    return index

def search_similar(query_vec: np.ndarray, index: faiss.Index, top_k=3):
    query_vec = np.expand_dims(query_vec, axis=0).astype("float32")
    # For IndexFlatIP, higher scores mean more similar (inner product)
    scores, indices = index.search(query_vec, top_k)
    return scores[0], indices[0]

# Load database PR extract files
import os
import glob

# Get all PR extract files from the database directory
db_pr_files = glob.glob('pr-extracts/pr-extract-*.yml')
db_pr_data = []
db_pr_texts = []
db_pr_ids = []

print(f"Loading {len(db_pr_files)} database PR extract files...")

# Load each database PR extract file
for file_path in db_pr_files:
    with open(file_path, 'r') as file:
        pr = yaml.safe_load(file)
        db_pr_data.append(pr)
        pr_text = yaml_to_text(pr)
        db_pr_texts.append(pr_text)
        db_pr_ids.append(pr['pr_id'])
        # store anchors
        pr.setdefault('_anchors', extract_anchors(pr))
        print(f"Loaded {pr['pr_id']} from {os.path.basename(file_path)}")

# Create embeddings for each database PR
print("\nCreating embeddings for database PRs...")
db_pr_embeddings_raw = []
for i, text in enumerate(db_pr_texts):
    print(f"Embedding {db_pr_ids[i]}...")
    embedding = embed_text(text)
    db_pr_embeddings_raw.append(embedding)

# Center embeddings by subtracting corpus mean and then normalizing
db_corpus_mean = np.mean(db_pr_embeddings_raw, axis=0)
db_pr_embeddings = []
for vec in db_pr_embeddings_raw:
    adj = vec - db_corpus_mean
    norm = np.linalg.norm(adj)
    if norm > 0:
        adj = adj / norm
    db_pr_embeddings.append(adj)

# Build FAISS index for database PRs
print("\nBuilding FAISS index for database PRs...")
db_index = build_faiss_index(db_pr_embeddings)

# Load query PR from separate directory
query_file_path = 'pr-query/pr-extract-006.yml'
print(f"\nLoading query PR from {query_file_path}...")
with open(query_file_path, 'r') as file:
    query_pr = yaml.safe_load(file)
    query_pr_id = query_pr['pr_id']
    query_pr_text = yaml_to_text(query_pr)
    query_pr['_anchors'] = extract_anchors(query_pr)

# Create embedding for query PR
print(f"Creating embedding for query PR {query_pr_id}...")
query_vec_raw = embed_text(query_pr_text)

# Apply same transformation as database PRs
query_vec = query_vec_raw - db_corpus_mean
qnorm = np.linalg.norm(query_vec)
if qnorm > 0:
    query_vec = query_vec / qnorm

# Run similarity search
print(f"\nRunning similarity search with {query_pr_id} as the query...")
top_k = min(3, len(db_pr_embeddings))  # Get top 3 or fewer if we have fewer PRs
distances, indices = search_similar(query_vec, db_index, top_k=top_k)

# Show results
print("\nSimilarity Search Results (Embedding Only):")
print("-" * 50)
print(f"Query PR: {query_pr_id}")
print("-" * 50)
for rank, (score, idx) in enumerate(zip(distances, indices)):
    if idx >= 0 and idx < len(db_pr_ids):
        similarity_percentage = (score + 1) / 2 * 100  # map [-1,1] -> [0,100]
        print(f"Rank {rank+1}: {db_pr_ids[idx]}, Similarity: {similarity_percentage:.2f}%")
    else:
        print(f"Rank {rank+1}: Invalid index {idx}")

# Calculate combined scores with anchor similarity
combined = []
for score, idx in zip(distances, indices):
    if 0 <= idx < len(db_pr_ids):
        # Use hierarchical similarity between query PR and each database PR
        anchor_sim = hierarchical_similarity(query_pr['_anchors'], db_pr_data[idx]['_anchors'])
        # Adjust weights: 30% embedding, 70% anchor similarity
        # This gives more weight to structural similarity than semantic similarity
        combined_score = 0.3 * score + 0.7 * anchor_sim
        combined.append((combined_score, idx, score, anchor_sim))

# Re-rank by combined_score
combined.sort(key=lambda x: x[0], reverse=True)

print("\nCombined Similarity Results (Embedding 30% + Anchor 70%):")
print("-" * 50)
for rank,(cscore, idx, escore, asim) in enumerate(combined[:top_k]):
    similarity_pct = (cscore + 1) / 2 * 100
    print(f"Rank {rank+1}: {db_pr_ids[idx]}, Combined: {similarity_pct:.2f}%  (emb {(escore+1)/2*100:.2f}%, anchor {asim*100:.2f}%)")

# Print details of top matches
print("\nDetails of top matches:")
print("-" * 50)
print(f"QUERY PR: {query_pr_id}")
print("File changes:")
for file in query_pr['file_changes']:
    print(f"  - {file['operation']} {file['filename']}")
print("-" * 50)

# Print details of top database matches
for cscore, idx, escore, asim in combined[:3]:  # Show top 3 matches
    similarity_pct = (cscore + 1) / 2 * 100
    print(f"PR ID: {db_pr_ids[idx]}")
    print(f"Combined Similarity: {similarity_pct:.2f}%  (emb {(escore+1)/2*100:.2f}%, anchor {asim*100:.2f}%)")
    if db_pr_ids[idx] == 'PR-7823':
        print("[AUTH PR MATCH]")
    print("File changes:")
    for file in db_pr_data[idx]['file_changes']:
        print(f"  - {file['operation']} {file['filename']}")
    print("-" * 50)
