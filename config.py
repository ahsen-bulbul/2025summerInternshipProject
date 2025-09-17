import torch
from dataclasses import dataclass

@dataclass
class Config:
    SPARSE_MODEL: str = "Qdrant/bm25"
    USE_FP16: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TOKEN_SIZE: int = 512
    ENCODING_NAME: str = "cl100k_base"
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "deneme"
    CSV_FILE: str = "/home/ahsen/Masaüstü/stajProjesi/2025summerInternshipProject/data/cleaned10chunk.csv"
    BATCH_SIZE: int = 100
    DB_BATCH: int = 256
    #model seçimine göre değişen özellikler
    model_name: str = "BAAI/bge-m3"
    model_type: str = "bge"
    embedding_dim: int = 512
    max_seq_length: int = 512
    description: str = "BGE-M3 - Çok dilli, dense+sparse embedding destekli"
    
@dataclass
class Models:
    bge_m3: Config
    bge_large: Config
    multilingual_e5: Config
    turkish_bert: Config
    distilbert_turkish: Config
    all_mpnet: Config


model = Models(
    bge_m3=Config(
        model_name="BAAI/bge-m3",
        model_type="bge",
        embedding_dim=1024,
        max_seq_length=8192,
        description="BGE-M3 - Çok dilli, dense+sparse embedding destekli"
    ),
    bge_large=Config(
        model_name="BAAI/bge-large-en-v1.5",
        model_type="sentence_transformer",
        embedding_dim=1024,
        max_seq_length=512,
        description="BGE Large - Sadece dense embedding"
    ),
    multilingual_e5=Config(
        model_name="intfloat/multilingual-e5-large",
        model_type="sentence_transformer",
        embedding_dim=1024,
        max_seq_length=512,
        description="E5 Multilingual Large - Çok dilli dense embedding"
    ),
    turkish_bert=Config(
        model_name="dbmdz/bert-base-turkish-cased",
        model_type="sentence_transformer",
        embedding_dim=768,
        max_seq_length=512,
        description="Turkish BERT - Türkçe özelleştirilmiş"
    ),
    distilbert_turkish=Config(
        model_name="dbmdz/distilbert-base-turkish-cased",
        model_type="sentence_transformer",
        embedding_dim=768,
        max_seq_length=512,
        description="Hızlı Türkçe DistilBERT"
    ),
    all_mpnet=Config(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_type="sentence_transformer",
        embedding_dim=768,
        max_seq_length=384,
        description="All-MiniLM - Genel amaçlı, hızlı"
    )
)