import pandas as pd
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tiktoken

@dataclass
class ChunkMetadata:
    """Chunk metadata sınıfı"""
    document_id: str
    location: str
    section_type: str
    section_title: str
    chunk_index: int
    total_chunks: int
    tokens: int
    characters: int
    has_dates: bool
    has_legal_refs: bool
    case_numbers: List[str]
    dates: List[str]

class TurkishLegalChunker:
    """Türkçe hukuki metinler için özelleştirilmiş chunker"""
    
    def __init__(self, 
                 target_tokens: int = 500, 
                 max_tokens: int = 800, 
                 min_tokens: int = 100,
                 overlap_ratio: float = 0.1):
        
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
        
        # Tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Regex patterns
        self.section_patterns = [
            r'([IVX]+\.\s+[A-ZÜĞŞÇÖIÜ][A-ZÜĞŞÇÖIÜa-züğşçöıi\s]+)',  # Roma rakamları
            r'([A-Z]\.\s+[A-ZÜĞŞÇÖIÜ][A-ZÜĞŞÇÖIÜa-züğşçöıi\s]*)',   # A. B. C. bölümler
            r'(\d+\.\s*[A-ZÜĞŞÇÖIÜ][A-ZÜĞŞÇÖIÜa-züğşçöıi\s]*)',    # 1. 2. 3. bölümler
        ]
        
        self.date_pattern = r'\d{1,2}\.\d{1,2}\.\d{4}'
        self.case_number_pattern = r'\d{4}/\d+\s+[EK]\.|[EK]\.\s*,\s*\d{4}/\d+\s+[EK]\.'
        
        # Türkçe karakter düzeltme haritası
        self.char_fixes = {
            'Ä°': 'İ', 'Åž': 'Ş', 'ÄŸ': 'ğ', 'Ã¼': 'ü', 'Ã§': 'ç', 
            'Ä±': 'ı', 'Ã¶': 'ö', 'Ã': 'İ', 'Ã‡': 'Ç', 'ÃœÃ‡': 'ÜÇ',
            'Â': '', 'â€': '', 'â€œ': '"', 'â€': '"'
        }
        
        # Hukuki terimler (bölüm tespiti için)
        self.legal_sections = {
            'DAVA', 'CEVAP', 'MAHKEME', 'KARAR', 'TEMYIZ', 'ISTINAF', 
            'BOZMA', 'GEREKÇE', 'DEĞERLENDIRME', 'SONUÇ'
        }

    def count_tokens(self, text: str) -> int:
        """Token sayısını hesapla"""
        return len(self.encoding.encode(text))
    
    def fix_encoding(self, text: str) -> str:
        """Türkçe karakter encoding sorunlarını düzelt"""
        for wrong, correct in self.char_fixes.items():
            text = text.replace(wrong, correct)
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Metni ön işlemden geçir"""
        # Encoding düzelt
        text = self.fix_encoding(text)
        
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Satır başlarındaki boşlukları temizle
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_metadata_info(self, text: str) -> Dict:
        """Metinden metadata bilgilerini çıkar"""
        # Tarihler
        dates = re.findall(self.date_pattern, text)
        
        # Dava numaraları
        case_numbers = re.findall(self.case_number_pattern, text)
        
        # Hukuki referanslar
        legal_refs = bool(re.search(r'\d+\s+sayılı|HMK|HUMK|İİK|TCK', text))
        
        return {
            'dates': list(set(dates)),
            'case_numbers': list(set(case_numbers)),
            'has_legal_refs': legal_refs,
            'has_dates': len(dates) > 0
        }
    
    def detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Metindeki bölümleri tespit et"""
        sections = []
        
        for pattern in self.section_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                section_title = match.group(1).strip()
                start = match.start()
                # Bir sonraki bölümü bul
                end = len(text)
                
                # Sonraki eşleşmeyi bul
                next_match = None
                for next_pattern in self.section_patterns:
                    next_matches = list(re.finditer(next_pattern, text[match.end():]))
                    if next_matches:
                        if next_match is None or next_matches[0].start() < next_match:
                            next_match = next_matches[0].start() + match.end()
                
                if next_match:
                    end = next_match
                
                sections.append((section_title, start, end))
        
        # Sırala ve çakışmaları temizle
        sections = sorted(sections, key=lambda x: x[1])
        cleaned_sections = []
        
        for i, (title, start, end) in enumerate(sections):
            if i == 0:
                cleaned_sections.append((title, start, end))
            else:
                prev_end = cleaned_sections[-1][2]
                if start >= prev_end:
                    cleaned_sections.append((title, start, end))
                else:
                    # Çakışma varsa önceki bölümün sonunu güncelle
                    cleaned_sections[-1] = (cleaned_sections[-1][0], cleaned_sections[-1][1], start)
                    cleaned_sections.append((title, start, end))
        
        return cleaned_sections
    
    def split_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Cümleler bazında metni böl"""
        # Türkçe için cümle sonu işaretleri
        sentence_endings = r'[.!?]+(?=\s+[A-ZÜĞŞÇÖIÜ]|\s*$)'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = f"{current_chunk} {sentence}".strip()
            
            if self.count_tokens(test_chunk) > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """Chunk'lar arası örtüşme oluştur"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # İlk chunk
                overlapped_chunks.append(chunk)
            else:
                # Önceki chunk'ın sonundan bir kısmını al
                prev_chunk = chunks[i-1]
                overlap_tokens = int(self.count_tokens(prev_chunk) * self.overlap_ratio)
                
                # Son cümleleri al (yaklaşık)
                prev_words = prev_chunk.split()
                overlap_words = prev_words[-overlap_tokens*2:] if len(prev_words) > overlap_tokens*2 else prev_words
                overlap_text = " ".join(overlap_words)
                
                overlapped_chunk = f"{overlap_text} {chunk}"
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk_document(self, document_id: str, location: str, raw_text: str) -> List[Dict]:
        """Ana chunking fonksiyonu"""
        # Ön işlem
        text = self.preprocess_text(raw_text)
        
        # Metadata bilgilerini çıkar
        metadata_info = self.extract_metadata_info(text)
        
        # Bölümleri tespit et
        sections = self.detect_sections(text)
        
        all_chunks = []
        
        if not sections:
            # Bölüm bulunamazsa tüm metni chunk'la
            chunks = self.split_by_sentences(text, self.max_tokens)
            chunks = self.create_overlapping_chunks(chunks)
            
            for i, chunk_text in enumerate(chunks):
                if self.count_tokens(chunk_text) >= self.min_tokens:
                    chunk_data = {
                        'text': chunk_text,
                        'metadata': ChunkMetadata(
                            document_id=document_id,
                            location=location,
                            section_type="FULL_DOCUMENT",
                            section_title="Tam Metin",
                            chunk_index=i,
                            total_chunks=len(chunks),
                            tokens=self.count_tokens(chunk_text),
                            characters=len(chunk_text),
                            **metadata_info
                        )
                    }
                    all_chunks.append(chunk_data)
        
        else:
            # Bölüm bazında chunk'la
            for section_title, start, end in sections:
                section_text = text[start:end].strip()
                
                if not section_text:
                    continue
                
                # Bölüm tipini belirle
                section_type = "OTHER"
                for legal_term in self.legal_sections:
                    if legal_term in section_title.upper():
                        section_type = legal_term
                        break
                
                if self.count_tokens(section_text) <= self.max_tokens:
                    # Küçük bölüm, aynen kullan
                    if self.count_tokens(section_text) >= self.min_tokens:
                        chunk_data = {
                            'text': section_text,
                            'metadata': ChunkMetadata(
                                document_id=document_id,
                                location=location,
                                section_type=section_type,
                                section_title=section_title,
                                chunk_index=0,
                                total_chunks=1,
                                tokens=self.count_tokens(section_text),
                                characters=len(section_text),
                                **metadata_info
                            )
                        }
                        all_chunks.append(chunk_data)
                else:
                    # Büyük bölüm, alt chunk'lara böl
                    section_chunks = self.split_by_sentences(section_text, self.max_tokens)
                    section_chunks = self.create_overlapping_chunks(section_chunks)
                    
                    for i, chunk_text in enumerate(section_chunks):
                        if self.count_tokens(chunk_text) >= self.min_tokens:
                            chunk_data = {
                                'text': chunk_text,
                                'metadata': ChunkMetadata(
                                    document_id=document_id,
                                    location=location,
                                    section_type=section_type,
                                    section_title=section_title,
                                    chunk_index=i,
                                    total_chunks=len(section_chunks),
                                    tokens=self.count_tokens(chunk_text),
                                    characters=len(chunk_text),
                                    **metadata_info
                                )
                            }
                            all_chunks.append(chunk_data)
        
        return all_chunks

def process_legal_csv(csv_path: str, output_path: str, 
                     target_tokens: int = 500, 
                     max_tokens: int = 800) -> None:
    """CSV dosyasını işle ve chunk'ları kaydet"""
    
    print(f"📖 CSV dosyası okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} belge bulundu")
    
    # Chunker'ı başlat
    chunker = TurkishLegalChunker(target_tokens=target_tokens, max_tokens=max_tokens)
    
    all_chunks = []
    
    # Progress tracking
    total_docs = len(df)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"🔄 İşlenen: {idx}/{total_docs} (%{idx/total_docs*100:.1f})")
        
        try:
            chunks = chunker.chunk_document(
                document_id=row['_id'],
                location=row['location'],
                raw_text=row['rawText']
            )
            
            # DataFrame için düzleştir
            for chunk in chunks:
                chunk_row = {
                    'chunk_id': f"{chunk['metadata'].document_id}_{chunk['metadata'].chunk_index}",
                    'document_id': chunk['metadata'].document_id,
                    'location': chunk['metadata'].location,
                    'section_type': chunk['metadata'].section_type,
                    'section_title': chunk['metadata'].section_title,
                    'chunk_index': chunk['metadata'].chunk_index,
                    'total_chunks': chunk['metadata'].total_chunks,
                    'text': chunk['text'],
                    'tokens': chunk['metadata'].tokens,
                    'characters': chunk['metadata'].characters,
                    'has_dates': chunk['metadata'].has_dates,
                    'has_legal_refs': chunk['metadata'].has_legal_refs,
                    'dates': json.dumps(chunk['metadata'].dates),
                    'case_numbers': json.dumps(chunk['metadata'].case_numbers)
                }
                all_chunks.append(chunk_row)
                
        except Exception as e:
            print(f"❌ Hata - Belge {row['_id']}: {str(e)}")
            continue
    
    print(f"✅ Toplam {len(all_chunks)} chunk oluşturuldu")
    
    # Sonuçları kaydet
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # İstatistikler
    print("\n📊 İstatistikler:")
    print(f"Ortalama chunk boyutu: {chunks_df['tokens'].mean():.1f} token")
    print(f"Medyan chunk boyutu: {chunks_df['tokens'].median():.1f} token")
    print(f"Min chunk boyutu: {chunks_df['tokens'].min()} token")
    print(f"Max chunk boyutu: {chunks_df['tokens'].max()} token")
    print(f"Tarih içeren chunk'lar: {chunks_df['has_dates'].sum()}")
    print(f"Hukuki referans içeren chunk'lar: {chunks_df['has_legal_refs'].sum()}")
    
    # Bölüm türleri
    print(f"\n🏷️ Bölüm türleri:")
    print(chunks_df['section_type'].value_counts())
    
    print(f"\n💾 Sonuçlar kaydedildi: {output_path}")

# Örnek kullanım
if __name__ == "__main__":
    # CSV'yi işle
    process_legal_csv(
        csv_path='yargitay.csv',
        output_path='legal_chunks.csv',
        target_tokens=500,
        max_tokens=800
    )
    
    print("🎉 İşlem tamamlandı!")
    
    # Sonuçları kontrol et
    df = pd.read_csv('legal_chunks.csv')
    print(f"\n📋 Örnek chunk'lar:")
    print(df[['chunk_id', 'section_type', 'tokens', 'text']].head().to_string())