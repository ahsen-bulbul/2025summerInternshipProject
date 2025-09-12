import pandas as pd
import torch
import uuid
import json

from typing import List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, PointStruct, SearchRequest, SparseVectorParams
from qdrant_client.http.models import NamedVector, NamedSparseVector, SparseVector, SearchRequest
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken
import semchunk
from FlagEmbedding import BGEM3FlagModel
from config import Config
import os
from dotenv import load_dotenv

print(load_dotenv("/home/yapayzeka/ahsen_bulbul/qdrant/.env"))

def l2_normalize_tensor(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if t.dim() == 1:
        norm = torch.norm(t).clamp(min=eps)
        return t / norm
    norm = torch.norm(t, dim=1, keepdim=True).clamp(min=eps)
    return t / norm

class YargitaySemanticProcessor:
    def __init__(self, config: Config):
        self.config = config

        # Encoding & chunker
        self.encoding = tiktoken.get_encoding(config.ENCODING_NAME)
        self.chunker = semchunk.chunkerify(self.encoding, config.TOKEN_SIZE)

        # Model
        print(f"🔮 BGE-M3 yükleniyor: {config.BGE_MODEL_NAME} (device={config.DEVICE})")
        self.bge_model = BGEM3FlagModel(config.BGE_MODEL_NAME, use_fp16=config.USE_FP16, device=config.DEVICE)

        # Qdrant
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)

        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        print(f"✅ Hazır - Cihaz: {device_name}")

    # Test connection & print dense+sparse
    def test_bge_connection(self):
        try:
            test_text = ["Yargıtay 6. Hukuk Dairesi'nin ihtiyati tedbir kararı"]
            emb_res = self.bge_model.encode(test_text)
            dense = emb_res['dense_vecs'][0] if isinstance(emb_res, dict) and 'dense_vecs' in emb_res else emb_res[0]
            sparse_available = 'colbert_vecs' in emb_res
            print(f"✅ Dense embedding boyutu: {len(dense)}")
            print(f"🔍 Sparse embedding mevcut: {sparse_available}")
            return len(dense)
        except Exception as e:
            print(f"❌ BGE-M3 bağlantı hatası: {e}")
            return None

    def create_qdrant_collection(self, recreate: bool = False):
        collection_name = self.config.COLLECTION_NAME
        if recreate:
            try:
                self.qdrant_client.delete_collection(collection_name)
                print(f"🗑️ Eski koleksiyon silindi: {collection_name}")
            except Exception:
                pass

        try:
            existing = [c.name for c in self.qdrant_client.get_collections().collections]
            if collection_name not in existing:
                # Dense + Sparse (sparse için yine 512 dim)
                vectors_config = {
                    "dense_vec": models.VectorParams(size=self.config.EMBEDDING_DIM, distance=models.Distance.COSINE),
                }
                sparse_config = {
                    "sparse_vec": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False))
                }
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config = sparse_config
                )
                print(f"✅ Koleksiyon oluşturuldu: {collection_name} (Dense+Sparse)")
            else:
                print(f"ℹ️ Koleksiyon zaten var: {collection_name}")
        except Exception as e:
            print(f"❌ Koleksiyon oluşturma hatası: {e}")
            raise

    def semantic_chunk_text(self, text: str, metadata: dict = None) -> List[Dict]:
        if not text or not text.strip():
            return []
        try:
            chunks = self.chunker(text)
            result = []
            for i, c in enumerate(chunks):
                if c.strip():
                    cd = {
                        'chunk_id': i,
                        'text': c.strip(),
                        'token_count': len(self.encoding.encode(c)),
                        'char_count': len(c)
                    }
                    if metadata:
                        cd.update(metadata)
                    result.append(cd)
            return result
        except Exception as e:
            print(f"❌ Chunking hatası: {e}")
            return []
        
    def process_csv_file(self, csv_path: str) -> List[Dict]:
        print(f"📄 CSV okunuyor: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 {len(df)} satır yüklendi")
        except Exception as e:
            print(f"❌ CSV okuma hatası: {e}")
            return []

        text_column = next((c for c in ['rawText', 'chunk_text', 'text', 'content', 'metin'] if c in df.columns), None)
        if not text_column:
            print("❌ Ana metin sütunu bulunamadı")
            return []

        all_chunks = []
        for idx, row in df.iterrows():
            text = row.get(text_column, '')
            if not text or pd.isna(text):
                continue
            meta = {
                'original_index': idx,
                'esas_no': row.get('esasNo', ''),
                'karar_no': row.get('kararNo', ''),
                'daire': row.get('location', ''),
                'tarih': row.get('extractedDates', ''),
                'document_id': row.get('_id', ''),
            }
            chunks = self.semantic_chunk_text(str(text), meta)
            all_chunks.extend(chunks)
            if (idx+1)%5==0:
                print(f"  ✅ İşlenen satır: {idx+1}/{len(df)} (Toplam chunk: {len(all_chunks)})")

        print(f"🧩 Toplam {len(all_chunks)} chunk oluşturuldu")
        return all_chunks
    
    def create_embeddings_bge(self, texts: List[str], batch_size: int = None):
        batch_size = batch_size or self.config.BATCH_SIZE
        all_embeddings_dense, all_embeddings_sparse = [], []
        total = len(texts)
        print(f"🔮 {total} metin işleniyor (batch_size={batch_size})...")

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                # Model dense embedding üret
                emb_res = self.bge_model.encode(
                    batch_texts,
                    return_dense=True,
                    return_sparse=True
                )
                dense = emb_res.get("dense_vecs", [[0.0]*self.config.EMBEDDING_DIM for _ in batch_texts])

                # Dense içinde None veya kısa vektör varsa düzelt
                dense_clean = []
                for vec in dense:
                    if vec is None:
                        dense_clean.append([0.0]*self.config.EMBEDDING_DIM)
                    elif len(vec) < self.config.EMBEDDING_DIM:
                        dense_clean.append(vec + [0.0]*(self.config.EMBEDDING_DIM - len(vec)))
                    else:
                        dense_clean.append(vec[:self.config.EMBEDDING_DIM])

                # TF-IDF ile sparse embedding üret
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=5000)
                X_sparse = vectorizer.fit_transform(batch_texts)
                sparse_vectors = []
                for row in X_sparse:
                    row_coo = row.tocoo()
                    sparse_vectors.append({"indices": row_coo.col.tolist(), "values": row_coo.data.tolist()})

                # Listeye ekle
                all_embeddings_dense.extend(dense_clean)
                all_embeddings_sparse.extend(sparse_vectors)

                print(f"  📊 Batch işlendi: {i + len(batch_texts)}/{total}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ Embedding hatası (batch {i//batch_size+1}): {e}")
                all_embeddings_dense.extend([[0.0]*self.config.EMBEDDING_DIM for _ in batch_texts])
                all_embeddings_sparse.extend([{"indices": [], "values": []} for _ in batch_texts])

        return all_embeddings_dense, all_embeddings_sparse

    def upload_to_qdrant(self, chunks: List[Dict]):
        if not chunks:
            print("❌ Yüklenecek chunk yok")
            return

        print(f"🚀 {len(chunks)} chunk Qdrant'a yükleniyor...")
        texts = [c['text'] for c in chunks]
        embeddings_dense, embeddings_sparse = self.create_embeddings_bge(texts)

        points = []
        
        for c, d, s in zip(chunks, embeddings_dense, embeddings_sparse):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense_vec": d,
                    "sparse_vec": SparseVector(
                        indices=s["indices"],
                        values=s["values"]
                    )
                },
                payload=c,
            ))


        batch = self.config.DB_BATCH
        for i in range(0, len(points), batch):
            try:
                self.qdrant_client.upsert(collection_name=self.config.COLLECTION_NAME, points=points[i:i+batch])
                print(f"  ✅ Batch yüklendi: {min(i+batch,len(points))}/{len(points)}")
            except Exception as e:
                print(f"❌ Batch yükleme hatası: {e}")

        print("🎉 Yükleme tamamlandı!")
    
    def get_collection_info(self):
        try:
            info = self.qdrant_client.get_collection(self.config.COLLECTION_NAME)
            return {
                "collection_name": self.config.COLLECTION_NAME,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "embedding_model": "BGE-M3",
                "embedding_dim": self.config.EMBEDDING_DIM
            }
        except Exception as e:
            return {"error": str(e)}

    def search_semantic(self, query: str, limit: int = 10, score_threshold: float = None) -> List[Dict]:
        """
        Dense-only semantic search
        """
        try:
            emb_res = self.bge_model.encode([query])
            q_dense = emb_res['dense_vecs'] if isinstance(emb_res, dict) and 'dense_vecs' in emb_res else emb_res

            # Tensor -> first 512 dims -> list
            q_t = torch.tensor(q_dense, dtype=torch.float32, device=self.config.DEVICE)
            q_sliced = q_t[0, :self.config.EMBEDDING_DIM]
            query_v = NamedVector(
                name="dense_vec",
                vector=q_sliced.cpu().tolist()
            )

            qr = self.qdrant_client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=query_v,
                limit=limit,
                with_payload=True,
                #vector_name="dense_vec",
                score_threshold=score_threshold
            )

            results = [{"score": p.score, "payload": p.payload} for p in qr]
            print(f"📊 {len(results)} sonuç bulundu (Dense only)")
            return results

        except Exception as e:
            print(f"❌ Semantic search hatası: {e}")
            return []


    def search_hybrid(self, query: str, limit: int = 10, score_threshold: float = None) -> List[Dict]:
        """
        Dense + Sparse (TF-IDF) hybrid search
        """
        try:
            # --- Dense tarafı (BGE embeddings) ---
            emb_res = self.bge_model.encode(
                [query],
                return_dense=True,
                return_sparse=True
            )

            # Dense vektör
            q_dense = emb_res.get("dense_vecs", [[0.0]*self.config.EMBEDDING_DIM])[0]
            q_dense = q_dense[:self.config.EMBEDDING_DIM]  # boyut kırpma
            query_dense = NamedVector(
                name="dense_vec",
                vector=q_dense
            )

            # Sparse vektör (senin TF-IDF çıktın)
            query_sparse = None
            sparse_raw = emb_res.get("sparse_vecs", [None])[0]
            if sparse_raw and "indices" in sparse_raw and "values" in sparse_raw:
                query_sparse = NamedSparseVector(
                    name="sparse_vec",
                    vector=SparseVector(
                        indices=sparse_raw["indices"],
                        values=sparse_raw["values"]
                    )
                )

            # --- Search requests ---
            requests = [SearchRequest(vector=query_dense, limit=limit, with_payload=True, score_threshold=score_threshold)]
            if query_sparse:
                requests.append(SearchRequest(vector=query_sparse, limit=limit, score_threshold=score_threshold))

            qr = self.qdrant_client.search_batch(
                collection_name=self.config.COLLECTION_NAME,
                requests=requests,
            )

            # --- Sonuçları topla ---
            results = []
            for request_result in qr:  # her request_result: List[ScoredPoint]
                for scored_point in request_result:
                    results.append({
                        "score": scored_point.score,
                        "payload": scored_point.payload
                    })

            print(f"📊 {len(results)} sonuç bulundu (Dense + TF-IDF Sparse)")
            return results

        except Exception as e:
            print(f"❌ Hybrid search hatası: {e}")
            return []