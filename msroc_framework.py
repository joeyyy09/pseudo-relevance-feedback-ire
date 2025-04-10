import pandas as pd
import nltk
import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text_bm25(text):
    """
    Preprocesses text for BM25: lowercase, tokenize, remove punctuation,
    remove stopwords, remove URLs/mentions.
    """
    if not isinstance(text, str):
        return []
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ mentions
    text = re.sub(r'\@\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return processed_tokens

def preprocess_text_sbert(text):
    """
    Minimal preprocessing for Sentence-BERT: basic cleaning like URL removal.
    Keeping case might be beneficial for SBERT.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ mentions (optional, depends if they add noise or context)
    # text = re.sub(r'\@\w+', '', text)
    # Basic whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Data Loading ---
def load_articles(directory):
    """
    Loads news articles from files in a directory.
    Parses files where articles are concatenated and separated by '**************'.
    Each article has a header like '-----Title----- Date Time'.
    """
    articles = {} # article_id -> {title: ..., date_time: ..., text: ...}
    print(f"Loading articles from: {directory}")
    try:
        for filename in os.listdir(directory):
            if filename.startswith('.'): # Skip hidden files like .DS_Store
                continue
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"  Processing file: {filename}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    raw_articles = content.split('**************')
                    for i, raw_article in enumerate(raw_articles):
                        raw_article = raw_article.strip()
                        if not raw_article:
                            continue
                        
                        parts = raw_article.split('-----', 3)
                        if len(parts) >= 3:
                            header = parts[1].strip()
                            body = parts[2].strip() # Changed index from 3 to 2
                            
                            # Attempt to split header into title and date/time
                            # Assuming date/time is at the end, find the last occurrence of common month names
                            # This is heuristic and might need refinement
                            title = header
                            date_time_str = None
                            # Simple split based on the last space before potential date info - might be fragile
                            last_space_index = header.rfind('  ') # Look for double space often seen in example
                            if last_space_index != -1:
                                title = header[:last_space_index].strip()
                                date_time_str = header[last_space_index:].strip()
                            
                            article_id = f"{filename}_{i}"
                            articles[article_id] = {
                                'title': title,
                                'date_time': date_time_str, # Keep as string for now
                                'text': body
                            }
                        else:
                             print(f"    Warning: Could not parse article structure in {filename}, part {i}")
                             # Store raw content if parsing fails
                             article_id = f"{filename}_{i}_raw"
                             articles[article_id] = {'title': 'Unknown', 'date_time': 'Unknown', 'text': raw_article}

                except Exception as e:
                    print(f"    Error reading or parsing file {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: Directory not found - {directory}")
    except Exception as e:
        print(f"An unexpected error occurred while loading articles: {e}")
        
    print(f"Loaded {len(articles)} articles.")
    return articles

def load_tweets(filepath, chunksize=10000):
    """
    Loads tweets from a potentially large file with custom format.
    Fields are separated by "*,,,*".
    Each tweet contains:
    timestamp, tweet_id, text, user_id, username, hashtags, retweets, favorites
    """
    tweets = [] # list of dicts: {tweet_id: ..., text: ..., timestamp: ...}
    print(f"Loading tweets from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            count = 0
            batch = []
            for line in f:
                try:
                    # Split by the custom separator and clean up
                    parts = line.strip().split('*,,,*')
                    if len(parts) >= 3:  # Ensure we have at least timestamp, id, and text
                        timestamp = parts[0].strip('"')  # Remove quotes from timestamp
                        tweet_id = parts[1]
                        text = parts[2]
                        
                        tweet = {
                            'tweet_id': tweet_id,
                            'text': text,
                            'timestamp': timestamp
                        }
                        batch.append(tweet)
                        count += 1
                        
                        if len(batch) >= chunksize:
                            tweets.extend(batch)
                            print(f"  Processed {count} tweets so far")
                            batch = []
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
            
            # Add any remaining tweets
            if batch:
                tweets.extend(batch)

    except FileNotFoundError:
        print(f"Error: Tweet file not found - {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred while loading tweets: {e}")

    print(f"Loaded {len(tweets)} tweets.")
    return tweets


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting MSRoc Framework Execution...")

    # --- Parameters ---
    N_INITIAL_CANDIDATES = 50 # Top N tweets from BM25
    ALPHA_TERM_WEIGHTING = 0.5 # Weighting factor for term integration
    K_EXPANSION_TERMS = 10 # Number of terms to add for query expansion
    # MU_QUERY_EXPANSION = 0.8 # Weight for original vs expansion terms (Not directly used in simple concatenation)
    NEWS_DIR = 'IRE-dataset/IRENews'
    # For now, process tweets from a single day for demonstration
    # In a real scenario, you might loop through or combine tweet files
    TWEET_FILE = 'IRE-dataset/TweetsIRE/intweetsfile_Aug_01'  # Removed .csv extension

    # --- Load Data ---
    articles_data = load_articles(NEWS_DIR)
    # tweets_data is a list of dicts: [{'tweet_id': ..., 'text': ..., 'timestamp': ...}, ...]
    tweets_data = load_tweets(TWEET_FILE)

    if not articles_data or not tweets_data:
        print("Error: No articles or tweets loaded. Exiting.")
        exit()

    # --- Select Target Article (Example: first loaded article) ---
    target_article_id = list(articles_data.keys())[0]
    target_article_info = articles_data[target_article_id]
    target_article_text = target_article_info['text']
    print(f"\nTarget Article ID: {target_article_id}")
    print(f"Target Article Title: {target_article_info.get('title', 'N/A')}")

    # --- Preprocess Tweet Corpus for BM25 (Phase 0, Step 4) ---
    print("\nPreprocessing tweet corpus for BM25...")
    # Keep track of original tweet_id for mapping
    tweet_ids_ordered = [tweet['tweet_id'] for tweet in tweets_data]
    tokenized_tweets_bm25 = [preprocess_text_bm25(tweet['text']) for tweet in tweets_data]
    print(f"Preprocessed {len(tokenized_tweets_bm25)} tweets for BM25.")

    # --- Phase 1: Initial Retrieval (BM25) ---
    print("\n--- Phase 1: Initial Retrieval (BM25) ---")

    # Prepare Article Query (Step 6)
    print("Preprocessing target article for BM25 query...")
    article_query_tokens = preprocess_text_bm25(target_article_text)
    if not article_query_tokens:
         print("Warning: Target article produced no tokens after preprocessing for BM25.")
         # Handle case where article is empty or only stopwords/punctuation
         # Maybe use title? For now, we'll proceed but BM25 might perform poorly.
         # article_query_tokens = preprocess_text_bm25(target_article_info.get('title', ''))

    print(f"Article query tokens (sample): {article_query_tokens[:20]}...")

    # Index Tweets (Step 7)
    print("Initializing BM25 index...")
    bm25 = BM25Okapi(tokenized_tweets_bm25)
    print("BM25 index initialized.")

    # Get Initial Scores (Step 8)
    print("Calculating initial BM25 scores for all tweets...")
    initial_bm25_scores = bm25.get_scores(article_query_tokens)
    print(f"Calculated {len(initial_bm25_scores)} scores.")

    # Combine scores with tweet IDs
    # Ensure scores and IDs align correctly
    if len(initial_bm25_scores) != len(tweet_ids_ordered):
         print("Error: Mismatch between number of BM25 scores and number of tweets!")
         exit()
         
    scored_tweets = list(zip(tweet_ids_ordered, initial_bm25_scores))

    # Select Top N Tweets (Step 9)
    print(f"Selecting top {N_INITIAL_CANDIDATES} tweets based on BM25 scores...")
    # Sort by score descending
    scored_tweets.sort(key=lambda item: item[1], reverse=True)
    
    # Get the top N tweet IDs and their scores
    top_n_candidates = scored_tweets[:N_INITIAL_CANDIDATES]
    top_n_tweet_ids = [item[0] for item in top_n_candidates]
    initial_score_map = dict(top_n_candidates) # Store scores for later use {tweet_id: score}

    print(f"Selected {len(top_n_tweet_ids)} candidates.")
    print(f"Top 5 candidate IDs and scores: {top_n_candidates[:5]}")

    # --- Phase 2: Semantic Analysis (Sentence-BERT) ---
    print("\n--- Phase 2: Semantic Analysis (Sentence-BERT) ---")

    # Load Sentence-BERT Model (Step 10)
    print("Loading Sentence-BERT model ('all-MiniLM-L6-v2')...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence-BERT model loaded.")

    # Get Raw Texts for Top N Tweets (Step 11)
    # Create a lookup for faster access to tweet text by ID
    tweet_lookup = {tweet['tweet_id']: tweet['text'] for tweet in tweets_data}
    
    # Prepare texts for SBERT encoding (minimal preprocessing)
    print("Preparing texts for Sentence-BERT encoding...")
    article_text_sbert = preprocess_text_sbert(target_article_text)
    top_n_tweet_texts_sbert = [preprocess_text_sbert(tweet_lookup.get(tid, "")) for tid in top_n_tweet_ids]
    
    # Filter out any potential empty strings resulting from lookup failures or preprocessing
    valid_indices = [i for i, txt in enumerate(top_n_tweet_texts_sbert) if txt]
    valid_top_n_tweet_ids = [top_n_tweet_ids[i] for i in valid_indices]
    valid_top_n_tweet_texts_sbert = [top_n_tweet_texts_sbert[i] for i in valid_indices]

    if not article_text_sbert:
        print("Warning: Target article text is empty after SBERT preprocessing.")
        # Handle appropriately, maybe skip semantic analysis or use title
        article_embedding = np.zeros(sbert_model.get_sentence_embedding_dimension()) # Zero vector
    else:
        # Encode Texts (Step 12)
        print(f"Encoding target article...")
        article_embedding = sbert_model.encode(article_text_sbert, convert_to_tensor=False) # Get numpy array
        print("Article encoded.")

    if not valid_top_n_tweet_texts_sbert:
         print("Warning: No valid tweet texts found for semantic analysis after preprocessing.")
         top_n_tweet_embeddings = np.array([]) # Empty array
         semantic_scores = {}
    else:
        print(f"Encoding {len(valid_top_n_tweet_texts_sbert)} top N tweets...")
        top_n_tweet_embeddings = sbert_model.encode(valid_top_n_tweet_texts_sbert, convert_to_tensor=False)
        print("Top N tweets encoded.")

        # Calculate Semantic Similarities (Step 13)
        print("Calculating cosine similarities...")
        # Reshape article_embedding for pairwise calculation (1 sample vs N samples)
        similarities = cosine_similarity(article_embedding.reshape(1, -1), top_n_tweet_embeddings)[0] # Get the first row
        
        # Store similarities, mapping back to the original tweet IDs
        semantic_scores = {tid: sim for tid, sim in zip(valid_top_n_tweet_ids, similarities)}
        print(f"Calculated {len(semantic_scores)} semantic scores.")
        # Example: Print top 5 semantic scores
        sorted_semantic = sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True)
        print(f"Top 5 semantic scores: {sorted_semantic[:5]}")
        
    # Ensure semantic_scores covers all top_n_tweet_ids, potentially with 0 for those missing/empty
    full_semantic_scores = {tid: semantic_scores.get(tid, 0.0) for tid in top_n_tweet_ids}


    # --- Phase 3: Term Weighting and Integration ---
    print("\n--- Phase 3: Term Weighting and Integration ---")

    # Extract Terms from Top N Tweets (Step 14)
    print("Extracting terms from top N tweets (using BM25 preprocessing)...")
    top_n_processed_tweets = {} # {tweet_id: [token1, token2,...]}
    unique_terms = set()
    for tid in top_n_tweet_ids:
        original_text = tweet_lookup.get(tid)
        if original_text:
            processed = preprocess_text_bm25(original_text)
            top_n_processed_tweets[tid] = processed
            unique_terms.update(processed)
        else:
            top_n_processed_tweets[tid] = [] # Store empty list if text was missing

    print(f"Found {len(unique_terms)} unique terms in top {len(top_n_tweet_ids)} tweets.")

    # Calculate Integrated Term Weights (wt) (Step 15)
    print(f"Calculating integrated term weights (alpha={ALPHA_TERM_WEIGHTING})...")
    term_weights = {}
    processed_term_count = 0
    for term in unique_terms:
        containing_tweet_ids = [
            tid for tid, tokens in top_n_processed_tweets.items() if term in tokens
        ]

        if not containing_tweet_ids:
            continue # Should not happen based on unique_terms logic, but safe check

        # Get scores for tweets containing the term
        # Use initial_score_map for BM25 scores and full_semantic_scores for semantic scores
        term_bm25_scores = [initial_score_map.get(tid, 0.0) for tid in containing_tweet_ids]
        term_semantic_scores = [full_semantic_scores.get(tid, 0.0) for tid in containing_tweet_ids] # Use the full map

        # Calculate averages
        avg_bm25_score = np.mean(term_bm25_scores) if term_bm25_scores else 0.0
        avg_semantic_score = np.mean(term_semantic_scores) if term_semantic_scores else 0.0

        # Calculate integrated weight (wt) using Eq. 4 (simplified)
        wt = ALPHA_TERM_WEIGHTING * avg_bm25_score + (1 - ALPHA_TERM_WEIGHTING) * avg_semantic_score
        term_weights[term] = wt
        processed_term_count += 1
        if processed_term_count % 1000 == 0: # Print progress periodically
             print(f"  Processed {processed_term_count}/{len(unique_terms)} terms...")


    print(f"Calculated weights for {len(term_weights)} terms.")
    # Example: Print top 5 terms by weight
    sorted_term_weights = sorted(term_weights.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 terms by integrated weight: {sorted_term_weights[:5]}")

    # --- Phase 4: Query Expansion (Rocchio-like) ---
    print("\n--- Phase 4: Query Expansion ---")

    # Select Top K Expansion Terms (Step 16)
    # sorted_term_weights is already available from Phase 3
    expansion_terms = [term for term, weight in sorted_term_weights[:K_EXPANSION_TERMS]]
    print(f"Selected {len(expansion_terms)} expansion terms: {expansion_terms}")

    # Form Expanded Query (Q') (Step 17)
    # Simple concatenation as suggested for rank_bm25
    expanded_query_tokens = article_query_tokens + expansion_terms
    print(f"Original query length: {len(article_query_tokens)}, Expanded query length: {len(expanded_query_tokens)}")
    print(f"Expanded query tokens (sample): {expanded_query_tokens[:30]}...")

    # --- Phase 5: Second Retrieval (BM25 with Expanded Query) ---
    print("\n--- Phase 5: Second Retrieval (BM25 with Expanded Query) ---")

    # Get Final Scores (Step 18)
    # Use the same BM25 index initialized in Step 7
    print("Calculating final BM25 scores using the expanded query...")
    final_bm25_scores = bm25.get_scores(expanded_query_tokens)
    print(f"Calculated {len(final_bm25_scores)} final scores.")

    # Rank All Tweets (Step 19)
    # Combine final scores with tweet IDs
    if len(final_bm25_scores) != len(tweet_ids_ordered):
         print("Error: Mismatch between number of final BM25 scores and number of tweets!")
         # Decide how to handle: exit, or proceed with potentially misaligned data?
         # For now, let's try to proceed but warn heavily.
         min_len = min(len(final_bm25_scores), len(tweet_ids_ordered))
         final_scored_tweets = list(zip(tweet_ids_ordered[:min_len], final_bm25_scores[:min_len]))
         print(f"Warning: Proceeding with {min_len} tweets due to score/ID mismatch.")
    else:
        final_scored_tweets = list(zip(tweet_ids_ordered, final_bm25_scores))

    # Sort by final score descending
    final_scored_tweets.sort(key=lambda item: item[1], reverse=True)
    print("Ranked all tweets based on final scores.")

    # --- Phase 6: Output ---
    print("\n--- Phase 6: Final Output ---")
    
    # Present the final ranked list (Step 20)
    print(f"\nFinal Ranked Tweets for Article ID: {target_article_id}")
    print("--------------------------------------------------")
    # Print top 20 results as an example
    for rank, (tweet_id, score) in enumerate(final_scored_tweets[:20], 1):
        # Optionally retrieve and print tweet text for context
        tweet_text = tweet_lookup.get(tweet_id, "Tweet text not found.")
        # Limit text length for display
        display_text = (tweet_text[:100] + '...') if len(tweet_text) > 100 else tweet_text
        print(f"Rank {rank}: Tweet ID: {tweet_id}, Score: {score:.4f}")
        # print(f"   Text: {display_text}") # Uncomment to show text snippet

    print("\n--- MSRoc Framework Execution Complete ---")

    # Note: Phase 6 also mentioned modularization and parameter tuning,
    # which are good practices but not implemented in this single script run.

    # Process text
    doc = nlp("This is a sample text to process with spaCy")

    # Access the processed text
    for token in doc:
        print(token.text, token.pos_, token.dep_)
