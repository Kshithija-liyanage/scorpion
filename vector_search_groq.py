import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
import json
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = 'llama-3.3-70b-versatile'  # Free, fast, strong on Groq

assert all([PINECONE_API_KEY, PINECONE_INDEX, GROQ_API_KEY]), "Missing required environment variables."

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=False)

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False).tolist()

def upsert_texts(texts, ids=None):
    if ids is None:
        ids = [f"doc-{i}" for i in range(len(texts))]
    vectors = embed_texts(texts)
    to_upsert = list(zip(ids, vectors, texts))
    index.upsert(vectors=[(id, vec, {'text': txt}) for id, vec, txt in to_upsert])
    print(f"Upserted {len(texts)} texts.")

def search(query, top_k=3, min_score=0.8):
    print("Embedding query diff...")
    query_vec = embed_texts([query])[0]
    print(f"Searching for top {top_k} similar diffs with minimum score {min_score}...")
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    
    # Filter results by minimum score
    filtered_matches = [match for match in res['matches'] if match['score'] >= min_score]
    
    if not filtered_matches:
        print(f"No matches found with similarity score ≥ {min_score}.")
        return []
        
    print(f"\nFound {len(filtered_matches)} matches with similarity score ≥ {min_score}:")
    results = []
    bug_contexts = []
    
    for i, match in enumerate(filtered_matches):
        print(f"\n{'-'*50}")
        print(f"Match {i+1} (similarity score: {match['score']:.3f})")
        print(f"{'-'*50}")
        
        # Get the text for context
        diff_text = match['metadata'].get('text', '')
        metadata = match['metadata']
        
        # Check for bug information in metadata
        bug_ids = metadata.get('bug_ids', [])
        bug_titles = metadata.get('bug_titles', [])
        bug_descriptions = metadata.get('bug_descriptions', [])
        bug_rcas = metadata.get('bug_rcas', [])
        bug_count = len(bug_ids)
        
        # Create context with diff and bugs
        result_context = f"DIFF:\n{diff_text}\n\n"
        bug_context = ""
        
        if bug_count > 0:
            print(f"\n🐛 Found {bug_count} bugs in this diff:")
            bug_context += f"BUGS ({bug_count}):\n"
            
            for j in range(bug_count):
                bug_id = bug_ids[j] if j < len(bug_ids) else ''
                bug_title = bug_titles[j] if j < len(bug_titles) else ''
                bug_description = bug_descriptions[j] if j < len(bug_descriptions) else ''
                bug_rca = bug_rcas[j] if j < len(bug_rcas) else ''
                
                print(f"\n[{bug_id}] {bug_title}")
                print(f"Description: {bug_description}")
                print(f"Root Cause: {bug_rca}")
                
                bug_context += f"[{bug_id}] {bug_title}\n"
                bug_context += f"Description: {bug_description}\n"
                bug_context += f"Root Cause: {bug_rca}\n\n"
        else:
            print("\nNo bugs found in metadata for this diff.")
        
        result_context += bug_context
        results.append(result_context)
        bug_contexts.append(bug_context)
        
        # Print the full diff
        print("\nFull Diff:")
        print(diff_text)
    
    return results

def ask_groq(question, context):
    # Truncate context if it's too long (Groq has token limits)
    max_context_length = 20000  # Characters, not tokens
    if len(context) > max_context_length:
        print(f"Warning: Context too long ({len(context)} chars), truncating to {max_context_length} chars.")
        context = context[:max_context_length] + "..."
    
    prompt = f"""Analyze the following git diffs and their known bugs.\n\nGit Diffs and Known Bugs:\n{context}\n\nQuestion: {question}\n\nFocus ONLY on the specific bugs found in the metadata and the specific code in these diffs. Please:\n1. Analyze ONLY the specific bugs listed in the metadata\n2. Explain why these specific bugs occurred in this exact code\n3. Suggest precise fixes for these specific bugs\n4. Do NOT provide generic coding recommendations or best practices\n5. Do NOT mention issues that aren't directly related to the listed bugs\n6. Be concise and specific to these exact code changes and bugs"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful code review assistant specializing in identifying bugs and issues in git diffs."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1024,
        "top_p": 1
    }
    
    try:
        print("Sending request to Groq LLM...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as e:
        print(f"\nError from Groq API: {e}")
        if hasattr(response, 'text'):
            print(f"Response details: {response.text}")
        return f"Error: Failed to get response from Groq LLM. Check your API key and try again."
    except Exception as e:
        print(f"\nUnexpected error when calling Groq API: {e}")
        return f"Error: Failed to get response from Groq LLM. Details: {str(e)}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embed, search, and answer using Pinecone + Groq LLM.")
    parser.add_argument('--embed', nargs='+', help='Texts to embed and upsert to Pinecone.')
    parser.add_argument('--query', type=str, help='Query to search and answer.')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top matches to return (default: 3)')
    parser.add_argument('--min-score', type=float, default=0.8, help='Minimum similarity score threshold (default: 0.8)')
    parser.add_argument('--use-llm', action='store_true', help='Use Groq LLM to analyze the results')
    args = parser.parse_args()

    if args.embed:
        upsert_texts(args.embed)
    if args.query:
        results = search(args.query, top_k=args.top_k, min_score=args.min_score)
        if args.use_llm and results:
            context = '\n'.join(results)
            answer = ask_groq(args.query, context)
            print(f"\nLLM Analysis:\n{answer}")

if __name__ == "__main__":
    main()
