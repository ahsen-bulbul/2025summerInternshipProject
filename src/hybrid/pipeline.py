from processor import YargitaySemanticProcessor
from config import Config
import json

class YargitayPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.processor = YargitaySemanticProcessor(config)

    def full_pipeline(self, csv_path: str):
        self.processor.create_qdrant_collection(recreate=True)
        chunks = self.processor.process_csv_file(csv_path)
        self.processor.upload_to_qdrant(chunks)
        #info = self.processor.get_collection_info()
        #print(json.dumps(info, indent=2, ensure_ascii=False))

    def interactive_search(self):
        query = input("Arama metni: ").strip()
        results = self.processor.search_hybrid(query)
        for r in results[:5]:
            print(json.dumps(r, indent=2, ensure_ascii=False))

    def semantic(self):
        query = input("Arama metni: ").strip()
        results = self.processor.search_semantic(query)
        for r in results[:5]:
            print(json.dumps(r, indent=2, ensure_ascii=False))    