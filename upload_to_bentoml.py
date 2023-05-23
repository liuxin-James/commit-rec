import bentoml
from transformers import AutoTokenizer,AutoModel

model = AutoModel.from_pretrained("models/base-models/sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoModel.from_pretrained("models/base-models/sentence-transformers/all-MiniLM-L6-v2")

saved_model = bentoml.transformers.save_model("sbert",model)
saved_tokenizer = bentoml.transformers.save_model("sbert-tokenizer",tokenizer)

print(f"model saved:{saved_model},tokenizer saved:{saved_tokenizer}")
