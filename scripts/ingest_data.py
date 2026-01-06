"""Script to process and ingest data into the vector database.
Supports loading data from CSV files, chunking text, and storing in chroma vector database."""

import sys
import os
import pandas as pd
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import config
from backend.vectordb import vector_db


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
        )

    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def load_csv_data(file_path: str, text_column: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file

    Args:
        file_path: Path to CSV file
        text_column: Name of column containing text data

    Returns:
        List of dictionaries containing text and metadata
    """
    df = pd.read_csv(file_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}"
        )

    documents = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        metadata = {"source_row": int(idx), "text_column": text_column}

        # Add other columns as metadata
        for col in df.columns:
            if col != text_column:
                metadata[col] = str(row[col])

        documents.append({"text": text, "metadata": metadata})

    return documents


def process_and_ingest(
    documents: List[Dict[str, Any]], chunk_size: int = None, overlap: int = None
):
    """
    Process documents into chunks and ingest into vector database

    Args:
        documents: List of document dictionaries with 'text' and 'metadata' keys
        chunk_size: Chunk size override
        overlap: Overlap size override
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    all_chunks = []
    all_metadata = []

    print(f"Processing {len(documents)} documents...")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")

    for doc in documents:
        text = doc["text"]
        base_metadata = doc.get("metadata", {})

        # Create chunks
        chunks = chunk_text(text, chunk_size, overlap)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            # Create metadata for this chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["total_chunks"] = len(chunks)
            all_metadata.append(chunk_metadata)

    print(f"Created {len(all_chunks)} chunks")
    print("Ingesting into vector database...")

    # Reset collection and add documents.
    vector_db.create_collection(reset=True)
    vector_db.add_documents(all_chunks, all_metadata)

    print(f"Successfully ingested {len(all_chunks)} chunks")
    print(f"Collection info: {vector_db.get_collection_info()}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Ingest data into RAG system")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--column", required=True, help="Name of text column")
    parser.add_argument(
        "--chunk-size", type=int, help="Chunk size (default from config)"
    )
    parser.add_argument(
        "--overlap", type=int, help="Overlap size (default from config)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.file}...")
    documents = load_csv_data(args.file, args.column)

    # Process and ingest
    process_and_ingest(documents, chunk_size=args.chunk_size, overlap=args.overlap)

    print("Ingestion complete!")
