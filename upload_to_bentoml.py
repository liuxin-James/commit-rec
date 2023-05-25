import torch
import bentoml
from transformers import AutoTokenizer,AutoModel
from pytorch_widedeep.models import Wide, WideDeep
from models.rank_net import BertModel



wide = Wide(input_dim=6000,pred_dim=1)
bert_model = BertModel(freeze_bert=True)
widedeep = WideDeep(wide=wide,deeptext=bert_model)
device = "cuda" if torch.cuda.is_available() else "cpu"
widedeep.load_state_dict(torch.load("./models/trained/wd/wd_model.pt",map_location=device))

model = AutoModel.from_pretrained("models/base-models/sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoModel.from_pretrained("models/base-models/sentence-transformers/all-MiniLM-L6-v2")

saved_model = bentoml.transformers.save_model("sbert",model)
saved_tokenizer = bentoml.transformers.save_model("sbert-tokenizer",tokenizer)

print(f"model saved:{saved_model},tokenizer saved:{saved_tokenizer}")
