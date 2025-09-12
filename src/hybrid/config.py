import torch
from dataclasses import dataclass

@dataclass
class Config:
    BGE_MODEL_NAME: str = "BAAI/bge-m3"
    USE_FP16: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TOKEN_SIZE: int = 512
    ENCODING_NAME: str = "cl100k_base"
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "bge_hybrid_chunks"
    EMBEDDING_DIM: int = 512
    CSV_FILE: str = "/home/yapayzeka/ahsen_bulbul/data/cleaned10chunk.csv"
    BATCH_SIZE: int = 100
    DB_BATCH: int = 256

