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
    Preprocesses text for BM25: focuses on main content, preserves important terms,
    removes dates and noise, and tokenizes meaningfully.
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove dates and times (common in news articles)
    text = re.sub(r'\b\d{1,2}:\d{2}(?:am|pm)?\b', '', text)  # times
    text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b', '', text)  # dates
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', text)  # dates with slashes
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove common news article noise
    text = re.sub(r'\(reuters\)|\(ap\)|\(cnn\)|\(bbc\)', '', text)
    text = re.sub(r'by [a-z]+(?: [a-z]+)*', '', text)  # bylines
    
    # Remove punctuation except for hyphens in compound words
    text = re.sub(r'[^\w\s#-]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Custom stopwords for news articles
    news_stopwords = set(stop_words)
    news_stopwords.update(['said', 'says', 'according', 'reported', 'reports', 'told'])
    
    # Remove stopwords but keep important terms
    processed_tokens = []
    for word in tokens:
        if word.strip() and word not in news_stopwords:
            # Keep numbers if they're part of important context (like "zika virus")
            if word.isdigit() and len(word) > 4:  # Likely a year or significant number
                processed_tokens.append(word)
            elif not word.isdigit():  # Keep non-number words
                processed_tokens.append(word)
    
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
                        
                        # Clean up the text
                        text = text.replace('\\/', '/')  # Fix URL slashes
                        text = text.replace('\\"', '"')  # Fix quotes
                        text = text.replace('\\u2026', '...')  # Fix ellipsis
                        
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
    print("\nLoaded articles:")
    for article_id, article_info in articles_data.items():
        print(f"ID: {article_id}")
        print(f"Title: {article_info.get('title', 'N/A')}")
        print("---")

    # tweets_data is a list of dicts: [{'tweet_id': ..., 'text': ..., 'timestamp': ...}, ...]
    tweets_data = load_tweets(TWEET_FILE)

    if not articles_data or not tweets_data:
        print("Error: No articles or tweets loaded. Exiting.")
        exit()

    # --- Select Target Article (Zika Virus Article) ---
    # Find the Zika article by searching through titles
    zika_article_id = None
    for article_id, article_info in articles_data.items():
        if "Zika" in article_info.get('title', ''):
            zika_article_id = article_id
            break
            
    if not zika_article_id:
        print("Error: Could not find Zika virus article. Exiting.")
        exit()
        
    target_article_id = zika_article_id
    target_article_info = articles_data[target_article_id]
    # Use the title as the content for querying
    target_article_text = target_article_info['title']
    print(f"\nSelected Target Article:")
    print(f"ID: {target_article_id}")
    print(f"Title: {target_article_info.get('title', 'N/A')}")
    print(f"Content being used for query: {target_article_text}")

    # After loading the target article, let's print the actual content being used
    print("\nArticle content being used for query:")
    print(target_article_text[:500])  # Show first 500 chars
    print("\nPreprocessed tokens:")
    print(preprocess_text_bm25(target_article_text)[:50])  # Show first 50 tokens

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
         exit()

    print(f"Article query tokens (sample): {article_query_tokens[:200]}...")

    # Index Tweets (Step 7)
    print("Initializing BM25 index...")
    bm25 = BM25Okapi(tokenized_tweets_bm25)
    print("BM25 index initialized.")

    # Get Initial Scores (Step 8)
    print("Calculating initial BM25 scores for all tweets...")
    initial_bm25_scores = bm25.get_scores(article_query_tokens)
    print(f"Calculated {len(initial_bm25_scores)} scores.")

    # Combine scores with tweet IDs and texts
    scored_tweets = list(zip(tweet_ids_ordered, initial_bm25_scores, [tweet['text'] for tweet in tweets_data]))

    # Select Top N Tweets (Step 9)
    print(f"Selecting top {N_INITIAL_CANDIDATES} tweets based on BM25 scores...")
    scored_tweets.sort(key=lambda item: item[1], reverse=True)
    top_n_candidates = scored_tweets[:N_INITIAL_CANDIDATES]

    # Save initial results
    print("\nSaving initial BM25 results...")
    with open('initial_bm25_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Target Article: {target_article_info['title']}\n\n")
        f.write("Top N Tweets from BM25:\n")
        for i, (tweet_id, score, text) in enumerate(top_n_candidates, 1):
            f.write(f"\nRank {i} (Score: {score:.4f}):\n")
            f.write(f"Tweet ID: {tweet_id}\n")
            f.write(f"Text: {text}\n")
            f.write("-" * 80 + "\n")

    # --- Phase 2: Semantic Analysis (Sentence-BERT) ---
    print("\n--- Phase 2: Semantic Analysis (Sentence-BERT) ---")

    # Load Sentence-BERT Model (Step 10)
    print("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # Get Raw Texts for Top N Tweets (Step 11)
    top_n_tweet_texts = [text for _, _, text in top_n_candidates]
    article_text_sbert = preprocess_text_sbert(target_article_text)
    top_n_tweet_texts_sbert = [preprocess_text_sbert(text) for text in top_n_tweet_texts]

    # Encode Texts (Step 12)
    print("Encoding texts...")
    article_embedding = sbert_model.encode(article_text_sbert, convert_to_tensor=False)
    tweet_embeddings = sbert_model.encode(top_n_tweet_texts_sbert, convert_to_tensor=False)

    # Calculate Semantic Similarities (Step 13)
    print("Calculating semantic similarities...")
    similarities = cosine_similarity(article_embedding.reshape(1, -1), tweet_embeddings)[0]
    semantic_scores = {tweet_id: sim for (tweet_id, _, _), sim in zip(top_n_candidates, similarities)}

    # --- Phase 3: Term Weighting and Integration ---
    print("\n--- Phase 3: Term Weighting and Integration ---")

    # Extract Terms from Top N Tweets (Step 14)
    print("Extracting terms from top N tweets...")
    top_n_processed_tweets = {}
    unique_terms = set()
    for tweet_id, _, text in top_n_candidates:
        processed = preprocess_text_bm25(text)
        top_n_processed_tweets[tweet_id] = processed
        unique_terms.update(processed)

    # Calculate Integrated Term Weights (Step 15)
    print("Calculating integrated term weights...")
    term_weights = {}
    for term in unique_terms:
        containing_tweet_ids = [tid for tid, tokens in top_n_processed_tweets.items() if term in tokens]
        if not containing_tweet_ids:
            continue

        # Get scores for tweets containing the term
        term_bm25_scores = [score for tid, score, _ in top_n_candidates if tid in containing_tweet_ids]
        term_semantic_scores = [semantic_scores.get(tid, 0.0) for tid in containing_tweet_ids]

        # Calculate averages
        avg_bm25_score = np.mean(term_bm25_scores) if term_bm25_scores else 0.0
        avg_semantic_score = np.mean(term_semantic_scores) if term_semantic_scores else 0.0

        # Calculate integrated weight
        wt = ALPHA_TERM_WEIGHTING * avg_bm25_score + (1 - ALPHA_TERM_WEIGHTING) * avg_semantic_score
        term_weights[term] = wt

    # --- Phase 4: Query Expansion ---
    print("\n--- Phase 4: Query Expansion ---")

    # Select Top K Expansion Terms (Step 16)
    sorted_term_weights = sorted(term_weights.items(), key=lambda item: item[1], reverse=True)
    expansion_terms = [term for term, _ in sorted_term_weights[:K_EXPANSION_TERMS]]
    print(f"Selected expansion terms: {expansion_terms}")

    # Form Expanded Query (Step 17)
    expanded_query_tokens = article_query_tokens + expansion_terms

    # --- Phase 5: Second Retrieval ---
    print("\n--- Phase 5: Second Retrieval ---")

    # Get Final Scores (Step 18)
    print("Calculating final BM25 scores...")
    final_bm25_scores = bm25.get_scores(expanded_query_tokens)

    # Rank All Tweets (Step 19)
    final_scored_tweets = list(zip(tweet_ids_ordered, final_bm25_scores, [tweet['text'] for tweet in tweets_data]))
    final_scored_tweets.sort(key=lambda item: item[1], reverse=True)

    # Save final results
    print("\nSaving final results...")
    with open('final_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Target Article: {target_article_info['title']}\n\n")
        f.write("Final Ranked Tweets:\n")
        for i, (tweet_id, score, text) in enumerate(final_scored_tweets[:N_INITIAL_CANDIDATES], 1):
            f.write(f"\nRank {i} (Score: {score:.4f}):\n")
            f.write(f"Tweet ID: {tweet_id}\n")
            f.write(f"Text: {text}\n")
            f.write("-" * 80 + "\n")

    print("\nMSRoc Framework Execution Complete!")
    print("Results saved to 'initial_bm25_results.txt' and 'final_results.txt'")

    # Note: Phase 6 also mentioned modularization and parameter tuning,
    # which are good practices but not implemented in this single script run.

    # Process text
    doc = nlp("This is a sample text to process with spaCy")

    # Access the processed text
    for token in doc:
        print(token.text, token.pos_, token.dep_)