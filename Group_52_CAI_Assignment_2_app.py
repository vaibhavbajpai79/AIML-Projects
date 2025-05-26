"""SLM with RAG for financial statements"""

# Conversation AI (S1-24_AIMLCZG521) - Assignment 2
# Develop a Retrieval-Augmented Generation (RAG) model
# to answer financial questions based on company financial statements (last two years).
# Group – 52
# TEAM MEMBERS:
# 1. Adarsh S - 2023AA05811
# 2. Divakar Roy - 2023AA05721
# 3. Mugdha Hans - 2023AA05570
# 4. Sonika Bengani - 2023AA05522
# 5. Vaibhav Bajpai - 2023AA05631


# requirements for the application
# numpy
# pandas
# matplotlib
# torch
# transformers
# sentence_transformers
# spacy
# faiss-cpu
# pdfplumber
# rank_bm25
# fastapi
# gradio

# Importing the dependencies
import logging
import os
import subprocess
import time
import re
import pickle
import numpy as np
import pandas as pd
import torch
import spacy
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize logger
logging.basicConfig(
    # filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
os.makedirs("data", exist_ok=True)

# SLM: Microsoft PHI-2 model is loaded
# It does have higher memory and compute requirements compared to TinyLlama and Falcon
# But it gives the best results among the three
# DEVICE = "cpu"  # or cuda
DEVICE = "cuda"  # or cpu
# MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
# MODEL_NAME = "tiiuae/falcon-rw-1b"
MODEL_NAME = "microsoft/phi-2"
# MODEL_NAME = "google/gemma-3-1b-pt"
# Load the Tokenizer for PHI-2
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
MAX_TOKENS = tokenizer.model_max_length
CONTEXT_MULTIPLIER = 0.7
# The max_context tokens is used to limit the retrieved chunks during querying
# to provide some headroom for the query
MAX_CONTEXT_TOKENS = int(MAX_TOKENS * CONTEXT_MULTIPLIER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Since the model is to be hosted on a cpu instance, we use float32
# For GPU, we can use float16 or bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(DEVICE)
model.eval()
# model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
logger.info("Model loaded successfully.")
# Load Sentence Transformer for Embeddings and Cross Encoder for re-ranking
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
# Load spaCy English model for Named Entity Recognition (mainly for guardrail)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Define the regex patterns, words blacklist
restricted_patterns = [
    r"\b(?:cfo|ceo|cto|executive|director|manager|employee|staff|worker)\b.*\b(?:salary|compensation|bonus|pay|income)\b",
    r"\b(?:salary|compensation|bonus|pay|income)\b.*\b(?:cfo|ceo|cto|executive|director|manager|employee|staff|worker)\b",
    r"\b(?:acquisition|merger|buyout)\b.*\b(?:before|pre-announcement|leak|inside information)\b",
    r"\b(?:before|pre-announcement|leak|inside information)\b.*\b(?:acquisition|merger|buyout)\b",
    r"\b(?:stock price|share price|insider trading|buying shares)\b",
    r"\b(?:internal policy|data breach|security protocol|confidential|classified)\b",
    r"\b(?:password|access credentials|encryption key|secure key)\b",
    r"\b(?:social security number|ssn|passport number|credit card|bank account|tax id|tin|personal details)\b",
    r"\b(?:employee records|payroll|medical records|hr data|salary data|pii|personally identifiable information)\b",
    r"\b(?:cfo|ceo|cto|executive|director|manager|employee|staff|worker)\b.*\b(?:address|work location|home location|residence|personal contact|phone number|email|office location)\b",
]

restricted_topics = {
    "CEO salary",
    "CFO salary",
    "executive pay",
    "stock options",
    "compensation details",
    "classified financial data",
    "insider trading",
    "password",
    "login credentials",
    "HR complaints",
    "remuneration",
    "director salary",
    "financial package",
}

FINANCIAL_ENTITY_LABELS = {"MONEY", "PERCENT", "CARDINAL", "ORG"}

GENERAL_KNOWLEDGE_PATTERNS = [
    r"\b(?:capital of|where is|who is|when did|what is|history of|define|meaning of|synonym of|antonym of|explain|how does|why is)\b",
    r"\b(?:country|city|continent|leader|president|prime minister|language|currency|population|politics|war|anthem|flag|national animal|national bird|national flower|national sport|monarch|king|queen|ruler|army|military|constitution|government|laws|famous person|historical figure|famous landmark|ocean|mountain|river|lake|climate|weather|culture|tradition|festival|holiday|invention|discovery|science|technology|art|literature|music|religion|mythology|folklore|education|university|school|mathematics|physics|chemistry|biology|philosophy|astronomy|space|planet|star|galaxy|universe|health|medicine|disease|virus|bacteria|genetics|DNA|evolution|ecology|environment|pollution|wildlife|habitat|natural disaster|earthquake|volcano|tsunami|hurricane|storm|flood|drought)\b",
    r"\b(?:[A-Z][a-z]+(?:'s)?\s+(?:capital|president|prime minister|national animal|national bird|national flower|national sport|anthem|flag|currency|language|leader|government|constitution|laws|monarch|king|queen|army|military|famous person|historical figure|landmark|river|ocean|mountain|religion|festival|holiday))\b",
]


sensitive_terms = {
    "salary",
    "compensation",
    "income",
    "pay",
    "bonus",
    "earnings",
    "wages",
}

EXPLANATORY_PATTERNS = [
    r"\b(why|reason|cause|explanation|due to|because|factor|impact of|effect of|influence of|driven by)\b",
    r"\b(how did|what led to|what caused|why did|how was|contributing factor|explain)\b",
]

FINANCIAL_DATA_PATTERNS = (
    r"\b(\₹?\s?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\s*(million|billion|crore|lakh|%)"
    r"?|Rs\.?\s?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?)\b"
)

FINANCIAL_TERMS = {
    "income",
    "revenue",
    "profit",
    "dividend",
    "investment",
    "earnings",
    "turnover",
    "expenses",
    "assets",
    "liabilities",
    "capital",
    "cash",
    "EBITDA",
    "margin",
    "tax",
    "costs",
    "reserves",
    "equity",
    "debt",
    "interest",
    "valuation",
    "amortization",
    "depreciation",
    "returns",
    "funds",
    "shares",
    "stock",
    "pricing",
    "liquidity",
    "credit",
    "bond",
    "expense",
    "budget",
    "yield",
    "growth",
}


# Extract the yaer from the upload file's name if any
def extract_year_from_filename(filename):
    """Extract Year from Filename"""
    match = re.search(r"(\d{4})-(\d{4})", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4})", filename)
    return match.group(1) if match else "Unknown"


# Use PDFPlumber to extract the tables from the uploaded file
# Add the year column for context and create a dataframe
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF into a DataFrame"""
    all_tables = []
    report_year = extract_year_from_filename(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                df["year"] = report_year
                all_tables.append(df)
    return pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()


# Load the csv files directly using pandas into a dataframe
def load_csv(file_path):
    """Loads a CSV file into a DataFrame"""
    try:
        df = pd.read_csv(file_path)
        df["year"] = extract_year_from_filename(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


# Preprocess the dataframe - Replace null values and create text rows suitable for chunking
def clean_dataframe_text(df):
    """Clean and format PDF/CSV data"""
    df.fillna("", inplace=True)
    text_data = []
    for _, row in df.iterrows():
        parts = []
        if "year" in df.columns:
            parts.append(f"Year: {row['year']}")
        parts.extend([str(val).strip() for val in row if str(val).strip()])
        text_data.append(", ".join(parts))
    df["text"] = text_data
    return df[["text"]].replace("", np.nan).dropna()


# Chunk the text for retrival
# Different chunk sizes - 256,512,1024,2048 were tried and 512 worked the best for financial RAG
def chunk_text(text, chunk_size=512):
    """Apply Chunking on the text"""
    words = text.split()
    chunks, temp_chunk = [], []
    for word in words:
        if sum(len(w) for w in temp_chunk) + len(temp_chunk) + len(word) <= chunk_size:
            temp_chunk.append(word)
        else:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = [word]
    if temp_chunk:
        chunks.append(" ".join(temp_chunk))
    return chunks


# Uses regex to identify financial terms and ensure relevant data is only merged
def is_financial_text(text):
    """Detects financial data"""
    return bool(
        re.search(
            FINANCIAL_DATA_PATTERNS,
            text,
            re.IGNORECASE,
        )
    )


# Advanced RAG - Chunk Merging
# Uses a sentence transformer "all-MiniLM-L6-v2" to embed text chunks
# Stores embeddings in a FAISS vector database for similarity search
# BM25 is implemented alongside FAISS to improve retrieval
# Use FAISS Cosine Similarity index and merge only highly similar text chunks (>85%)
def merge_similar_chunks(chunks, similarity_threshold=0.85):
    """Merge similar chunks while preserving financial data structure"""
    if not chunks:
        return []
    # Encode chunks into embeddings
    embeddings = np.array(
        embed_model.encode(chunks, normalize_embeddings=True), dtype="float32"
    )
    # FAISS Cosine Similarity Index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # Get top-2 most similar chunks
    _, indices = index.search(embeddings, 2)
    merged_chunks = {}
    for i, idx in enumerate(indices[:, 1]):
        if i in merged_chunks or idx in merged_chunks:
            continue
        sim_score = np.dot(embeddings[i], embeddings[idx])
        # Ensure financial data isn't incorrectly merged
        if is_financial_text(chunks[i]) or is_financial_text(chunks[idx]):
            merged_chunks[i] = chunks[i]
            merged_chunks[idx] = chunks[idx]
            continue
        # Merge only if similarity is high and chunks are adjacent
        if sim_score > similarity_threshold and abs(i - idx) == 1:
            merged_chunks[i] = chunks[i] + " " + chunks[idx]
            merged_chunks[idx] = merged_chunks[i]
        else:
            merged_chunks[i] = chunks[i]
    return list(set(merged_chunks.values()))


# Handle for file upload button in UI
# Processes the uploaded files and generates the embeddings
# The FAISS embeddings and tokenized chunks are saved for retrieval
def process_files(files, chunk_size=512):
    """Process uploaded files and generate embeddings"""
    if not files:
        logger.warning("No files uploaded!")
        return "Please upload at least one PDF or CSV file."
    pdf_paths = [file.name for file in files if file.name.endswith(".pdf")]
    csv_paths = [file.name for file in files if file.name.endswith(".csv")]
    logger.info(f"Processing {len(pdf_paths)} PDFs and {len(csv_paths)} CSVs")
    df_list = []
    if pdf_paths:
        df_list.extend([extract_tables_from_pdf(pdf) for pdf in pdf_paths])
    for csv in csv_paths:
        df = load_csv(csv)
        df_list.append(df)
    if not df_list:
        logger.warning("No valid data found in the uploaded files")
        return "No valid data found in the uploaded files"
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(how="all", inplace=True)
    logger.info("Data extracted from the files")
    df_cleaned = clean_dataframe_text(df)
    df_cleaned["chunks"] = df_cleaned["text"].apply(lambda x: chunk_text(x, chunk_size))
    df_chunks = df_cleaned.explode("chunks").reset_index(drop=True)
    merged_chunks = merge_similar_chunks(df_chunks["chunks"].tolist())
    chunk_texts = merged_chunks
    # chunk_texts = df_chunks["chunks"].tolist()
    embeddings = np.array(
        embed_model.encode(chunk_texts, normalize_embeddings=True), dtype="float32"
    )
    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "data/faiss_index.bin")
    logger.info("FAISS index created and saved.")
    # Save BM25 index
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25_data = {"tokenized_chunks": tokenized_chunks, "chunk_texts": chunk_texts}
    logger.info("BM25 index created and saved.")
    with open("data/bm25_data.pkl", "wb") as f:
        pickle.dump(bm25_data, f)
    return "Files processed successfully! You can now query."


def contains_financial_entities(query):
    """Check if query contains financial entities"""
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ in FINANCIAL_ENTITY_LABELS:
            return True
    return False


def contains_geographical_entities(query):
    """Check if the query contains geographical entities"""
    doc = nlp(query)
    return any(ent.label_ == "GPE" for ent in doc.ents)


def contains_financial_terms(query):
    """Check if the query contains financial terms"""
    return any(term in query.lower() for term in FINANCIAL_TERMS)


def is_general_knowledge_query(query):
    """Check if query contains general knowledge"""
    query_lower = query.lower()
    for pattern in GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def get_latest_available_year(retrieved_chunks):
    """Extracts the latest available year from retrieved financial data"""
    years = set()
    year_pattern = r"\b(20\d{2})\b"
    for chunk in retrieved_chunks:
        years.update(map(int, re.findall(year_pattern, chunk)))
    return max(years) if years else 2024


def is_irrelevant_query(query):
    """Check if the query is not finance related"""
    # If the query is general knowledge and not finance-related
    if is_general_knowledge_query(query) and not contains_financial_terms(query):
        return True
    # If the query contains only geographical terms without financial entities
    if contains_geographical_entities(query) and not contains_financial_entities(query):
        return True
    return False


# Input guardrail implementation
# NER + Regex + List of terms used to filter irrelevant queries
# Regex is used to filter queries related to sensitive topics
# Uses spaCy model's Named Entity Recognition to filter queries for personal details
# Uses cosine similarity with the embedded query and sensitive topic vectors
# to filter out queries violating confidential/security rules (additional)
def is_query_allowed(query):
    """Checks if the query violates security or confidentiality rules"""
    if is_irrelevant_query(query):
        return False, "Query is not finance-related. Please ask a financial question."
    for pattern in restricted_patterns:
        if re.search(pattern, query.lower(), re.IGNORECASE):
            return False, "This query requests sensitive or confidential information."
    doc = nlp(query)
    # Check if there's a person entity and contains sensitive terms
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for token in ent.subtree:
                if token.text.lower() in sensitive_terms:
                    return (
                        False,
                        "Query contains personal salary information, which is restricted.",
                    )
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    topic_embeddings = embed_model.encode(
        list(restricted_topics), normalize_embeddings=True
    )
    # Check similarities between the restricted topics and the query
    similarities = np.dot(topic_embeddings, query_embedding)
    if np.max(similarities) > 0.85:
        return False, "This query requests sensitive or confidential information."
    return True, None


# Boosts the scores for texts containing financial terms
# This is useful during re-ranking
def boost_score(text, base_score, boost_factor=1.2):
    """Boost scores if the text contains financial terms"""
    if any(term in text.lower() for term in FINANCIAL_TERMS):
        return base_score * boost_factor
    return base_score


# Advanced RAG - Adaptive Retrieval
# FAISS embeddings are used to retrieve semantically similar chunks
# BM25 is used to retrieve relevant chunks based on the keywords (TF-IDF)
# FAISS and BM25 complement each other- similar matches and important exact matches
# The retrieved chunks are merged and sorted based on a lambda FAISS value
# if lambda FAISS is 0.6, weightage for retrieved FAISS chunks are 0.6 and 0.4 for BM25 chunks
# Cross encoder model ms-marco-MiniLM-L6-v2 is used for scoring and re-ranking the chunks
def hybrid_retrieve(query, chunk_texts, index, bm25, top_k=5, lambda_faiss=0.7):
    """Hybrid Retrieval with FAISS, BM25, Cross-Encoder & Financial Term Boosting"""
    # FAISS Retrieval
    query_embedding = np.array(
        [embed_model.encode(query, normalize_embeddings=True)], dtype="float32"
    )
    _, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [chunk_texts[idx] for idx in faiss_indices[0]]
    # BM25 Retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [chunk_texts[idx] for idx in bm25_top_indices]
    # Merge FAISS & BM25 Scores
    results = {}
    for entry in faiss_results:
        results[entry] = boost_score(entry, lambda_faiss)
    for entry in bm25_results:
        results[entry] = results.get(entry, 0) + boost_score(entry, (1 - lambda_faiss))
    # Rank initial results
    retrieved_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)
    retrieved_texts = [r[0] for r in retrieved_docs]
    # Cross-Encoder Re-Ranking
    query_text_pairs = [[query, text] for text in retrieved_texts]
    scores = cross_encoder.predict(query_text_pairs)
    ranked_indices = np.argsort(scores)[::-1]
    # Return top-ranked results
    final_results = [retrieved_texts[i] for i in ranked_indices[:top_k]]
    return final_results


def compute_entropy(logits):
    """Compute entropy from logits."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()


def contains_future_year(query, retrieved_chunks):
    """Detects if the query asks for future data beyond available reports"""
    latest_year = get_latest_available_year(retrieved_chunks)
    # Extract years from query
    future_years = set(map(int, re.findall(r"\b(20\d{2})\b", query)))
    return any(year > latest_year for year in future_years)


def is_explanatory_query(query):
    """Checks if the query requires an explanation rather than factual data"""
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in EXPLANATORY_PATTERNS)


# A confidence score is computed using FAISS and BM25 ranking
# FAISS: The similarity score between the response and the retrieved chunks are normalized
# BM25: The BM25 scores for the query and response combined tokens is normalized
# The mean of top token probability mean and 1-entropy score is the model_conf_signal
# FAISS, BM25 and the model_conf_signal are combined using a weighted sum
def compute_response_confidence(
    query,
    response,
    retrieved_chunks,
    bm25,
    model_conf_signal,
    lambda_faiss=0.6,
    lambda_conf=0.3,
    lambda_bm25=1.0,
    future_penalty=-0.3,
    explanation_penalty=-0.2,
):
    """Calculates a confidence score for the model response"""
    if not retrieved_chunks:
        return 0.0
    # Compute FAISS similarity
    retrieved_embedding = embed_model.encode(
        " ".join(retrieved_chunks), normalize_embeddings=True
    )
    response_embedding = embed_model.encode(response, normalize_embeddings=True)
    faiss_score = np.dot(retrieved_embedding, response_embedding)
    # Normalize the FAISS score
    normalized_faiss = (faiss_score + 1) / 2
    # Compute BM25 for combined query + response
    tokenized_combined = (query + " " + response).lower().split()
    bm25_scores = bm25.get_scores(tokenized_combined)
    # Normalize the BM25 score
    if bm25_scores.size > 0:
        bm25_score = np.mean(bm25_scores)
        min_bm25, max_bm25 = np.min(bm25_scores), np.max(bm25_scores)
        normalized_bm25 = (
            (bm25_score - min_bm25) / (max_bm25 - min_bm25 + 1e-6)
            if min_bm25 != max_bm25
            else 0
        )
        normalized_bm25 = max(0, min(1, normalized_bm25))
    else:
        normalized_bm25 = 0.0
    # Penalize if query contains future years
    future_penalty = -0.3 if contains_future_year(query, retrieved_chunks) else 0.0
    # Penalize if query is reasoning based
    explanation_penalty_value = (
        explanation_penalty if is_explanatory_query(query) else 0.0
    )
    logger.info(
        f"Faiss score: {normalized_faiss}, BM25: {normalized_bm25}\n"
        f"Mean Top Token + 1-Entropy Avg: {model_conf_signal}\n"
        f"Future penalty: {future_penalty}, Reasoning penalty: {explanation_penalty_value}"
    )
    # Weighted sum of all the normalized scores
    confidence_score = (
        lambda_faiss * normalized_faiss
        + model_conf_signal * lambda_conf
        + lambda_bm25 * normalized_bm25
        + future_penalty
        + explanation_penalty_value
    )
    return round(min(100, max(0, confidence_score.item() * 100)), 2)


# UI handle for query model button
# Loads the saved FAISS embeddings and tokenized chunks for BM25
# Check the query for any policy violation
# Retrieve similar texts using the RAG implementation
# Prompt the loaded SLM along with the retrieved texts and compute confidence score
def query_model(
    query,
    top_k=10,
    lambda_faiss=0.5,
    repetition_penalty=1.5,
    max_new_tokens=100,
    use_extraction=False,
):
    """Query function"""
    start_time = time.perf_counter()
    # Check if FAISS and BM25 indexes exist
    if not os.path.exists("data/faiss_index.bin") or not os.path.exists(
        "data/bm25_data.pkl"
    ):
        logger.error("No index found! Prompting user to upload PDFs.")
        return (
            "Index files not found! Please upload PDFs first to generate embeddings.",
            "Error",
        )
    allowed, reason = is_query_allowed(query)
    if not allowed:
        logger.error(f"Query Rejected: {reason}")
        return f"Query Rejected: {reason}", "Warning"
    logger.info(
        f"Received query: {query} | Top-K: {top_k}, "
        f"Lambda: {lambda_faiss}, Tokens: {max_new_tokens}"
    )
    # Load FAISS & BM25 Indexes
    index = faiss.read_index("data/faiss_index.bin")
    with open("data/bm25_data.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    # Restore tokenized chunks and metadata
    tokenized_chunks = bm25_data["tokenized_chunks"]
    chunk_texts = bm25_data["chunk_texts"]
    bm25 = BM25Okapi(tokenized_chunks)
    retrieved_chunks = hybrid_retrieve(
        query, chunk_texts, index, bm25, top_k=top_k, lambda_faiss=lambda_faiss
    )
    logger.info("Retrieved chunks")
    context = ""
    token_count = 0
    # context = "\n".join(retrieved_chunks)
    for chunk in retrieved_chunks:
        chunk_tokens = tokenizer(chunk, return_tensors="pt")["input_ids"].shape[1]
        if token_count + chunk_tokens < MAX_CONTEXT_TOKENS:
            context += chunk + "\n"
            token_count += chunk_tokens
        else:
            break
    prompt = (
        "You are a financial analyst. Answer financial queries concisely using only the numerical data "
        "explicitly present in the provided financial context:\n\n"
        f"{context}\n\n"
        "Use only the given financial data—do not assume, infer, or generate missing values."
        " Retain the original format of financial figures without conversion."
        " If the requested information is unavailable, respond with 'No relevant financial data available.'"
        " Provide a single-sentence answer without explanations, additional text, or multiple responses."
        f"\nQuery: {query}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
    inputs.pop("token_type_ids", None)
    logger.info("Generating output")
    input_len = inputs["input_ids"].shape[-1]
    logger.info(f"Input len: {input_len}")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            repetition_penalty=repetition_penalty,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        sequences = output["sequences"][0][input_len:]
    execution_time = time.perf_counter() - start_time
    logger.info(f"Query processed in {execution_time:.2f} seconds.")
    # Get the logits per generated token
    log_probs = output["scores"]
    token_probs = [torch.softmax(lp, dim=-1) for lp in log_probs]
    # Extract top token probabilities for each step
    token_confidences = [tp.max().item() for tp in token_probs]
    # Compute final confidence score
    top_token_conf = sum(token_confidences) / len(token_confidences)
    print(f"Token Token Probability Mean: {top_token_conf:.4f}")
    entropy_score = sum(compute_entropy(lp) for lp in log_probs) / len(log_probs)
    entropy_conf = 1 - (entropy_score / torch.log(torch.tensor(tokenizer.vocab_size)))
    print(f"Entropy-based Confidence: {entropy_conf:.4f}")
    model_conf_signal = (top_token_conf + (1 - entropy_conf)) / 2
    response = tokenizer.decode(sequences, skip_special_tokens=True)
    confidence_score = compute_response_confidence(
        query, response, retrieved_chunks, bm25, model_conf_signal
    )
    logger.info(f"Confidence: {confidence_score}%")
    if confidence_score <= 0.3:
        logger.error(f"The system is unsure about this response.")
        response += "\nThe system is unsure about this response."
    final_out = ""
    if not use_extraction:
        final_out += f"Context: {context}\nQuery: {query}\n"
    final_out += f"Response: {response}"
    return (
        final_out,
        f"Confidence: {confidence_score}%\nTime taken: {execution_time:.2f} seconds",
    )


# Gradio UI
with gr.Blocks(title="Financial Statement RAG with LLM") as ui:
    gr.Markdown("## Financial Statement RAG with LLM")
    # File upload section
    with gr.Group():
        gr.Markdown("###  Upload & Process Annual Reports")
        file_input = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".csv"],
            type="filepath",
            label="Upload Annual Reports (PDFs/CSVs)",
        )
        process_button = gr.Button("Process Files")
        process_output = gr.Textbox(label="Processing Status", interactive=False)
    # Query model section
    with gr.Group():
        gr.Markdown("###  Ask a Financial Query")
        query_input = gr.Textbox(label="Enter Query")
        with gr.Row():
            top_k_input = gr.Number(value=15, label="Top K (Default: 15)")
            lambda_faiss_input = gr.Slider(0, 1, value=0.5, label="Lambda FAISS (0-1)")
            repetition_penalty = gr.Slider(
                1, 2, value=1.2, label="Repetition Penality (1-2)"
            )
            max_tokens_input = gr.Number(value=100, label="Max New Tokens")
        use_extraction = gr.Checkbox(label="Retrieve only the answer", value=False)
        query_button = gr.Button("Submit Query")
        query_output = gr.Textbox(label="Query Response", interactive=False)
        time_output = gr.Textbox(label="Time Taken", interactive=False)
    # Button Actions
    process_button.click(process_files, inputs=[file_input], outputs=process_output)
    query_button.click(
        query_model,
        inputs=[
            query_input,
            top_k_input,
            lambda_faiss_input,
            repetition_penalty,
            max_tokens_input,
            use_extraction,
        ],
        outputs=[query_output, time_output],
    )

# Application entry point
if __name__ == "__main__":
    logger.info("Starting Gradio server...")
    ui.launch(server_name="0.0.0.0", server_port=7860, pwa=True)
