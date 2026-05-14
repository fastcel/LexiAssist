## LexiAssist

LexiAssist is a Retrieval-Augmented Generation (RAG) system for legal documents that answers users’ legal queries in plain English while providing citations from authentic legal sources.

The system performs the following tasks:

1. Indexes Australian legal documents from the Open Australian Legal Corpus
2. Accepts natural-language legal questions from users
3. Retrieves the most relevant document chunks using hybrid dense and sparse retrieval
4. Reranks retrieved results using a cross-encoder for improved precision
5. Generates plain-English responses with inline citations using GPT-4o
6. Provides a web-based GUI for chat, document browsing, and legal text simplification

We have implemented two versions of LexiAssist:
A version using pretrained models
A version using models trained/fine-tuned by us

Setup instructions for both versions are provided below..

## Setup instructions for both versions
1. Install dependencies
pip install -r requirements.txt

2. Build retrieval index 
python indexer.py

3. Start server
python server.py


