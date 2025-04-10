# MSRoc Framework for Tweet-Article Relevance

The MSRoc Framework is an advanced Information Retrieval system designed to find relevant tweets for news articles by combining traditional keyword-based retrieval (BM25) with modern semantic analysis (Sentence-BERT). This hybrid approach significantly improves the relevance of retrieved tweets compared to using either method alone.

## Project Overview

The framework was implemented to process a large dataset of tweets (3.8 million) and news articles (792) from August 2016, focusing on political news coverage. The system successfully demonstrated the effectiveness of combining lexical and semantic matching for tweet retrieval.

## Implementation Details

The framework consists of six main phases:

1. **Initial Retrieval (BM25)**
   - Preprocesses tweets and articles using NLTK and spaCy
   - Uses BM25 algorithm for initial keyword-based retrieval
   - Selects top 50 candidate tweets based on lexical similarity

2. **Semantic Analysis (Sentence-BERT)**
   - Employs the 'all-MiniLM-L6-v2' model for semantic encoding
   - Computes cosine similarities between article and tweets
   - Identifies semantically similar content beyond keyword matching

3. **Term Weighting and Integration**
   - Extracts key terms from top candidate tweets
   - Calculates integrated weights using both BM25 and semantic scores
   - Uses alpha=0.5 for balanced weighting between methods

4. **Query Expansion**
   - Automatically identifies relevant expansion terms
   - Adds top weighted terms to the original query
   - Example expansion: ['august', 'augustaugust', 'rt']

5. **Second Retrieval**
   - Re-runs BM25 with expanded query
   - Combines lexical and semantic relevance scores
   - Produces final ranking of all tweets

6. **Final Output**
   - Ranks tweets by integrated relevance score
   - Provides top 20 most relevant tweets
   - Includes tweet IDs and relevance scores

## Results

For the sample article "Clinton emails reveal plans to depict Ryan as anti-Trump", the framework:
- Successfully processed 3,809,428 tweets
- Achieved high relevance scores (top scores around 29.58)
- Demonstrated effective query expansion
- Produced a diverse set of relevant tweets

## Technical Requirements

- Python 3.11
- Dependencies listed in requirements.txt
- NLTK data and spaCy model
- Sufficient memory for processing large datasets

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Place data in the IRE-dataset directory:
```
IRE-dataset/
├── IRENews/          # News articles
└── TweetsIRE/        # Tweet files
```

3. Run the framework:
```bash
python msroc_framework.py
```

## Notes

- The framework is optimized for political news and tweets
- Processing time varies based on dataset size
- Memory usage scales with the number of tweets
- Results are saved to msroc_output.txt
