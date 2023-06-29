import torch
import pickle
import bentoml
from transformers import AutoTokenizer, AutoModel
from pytorch_widedeep.models import Wide, WideDeep
from models.rank_net import BertModel
from rec_models.models import RecNet

device = "cuda" if torch.cuda.is_available() else "cpu"


def upload_widedeep():
    # load wide_preprocess
    wide_preprocess = None
    with open("models/trained/wd/wide_preprocess.pkl", "rb") as f:
        wide_preprocess = pickle.load(f)

    # load widedeep component
    wide = Wide(input_dim=wide_preprocess.wide_dim, pred_dim=1)
    bert_model = BertModel(freeze_bert=True)

    # build widedeep model
    widedeep = WideDeep(wide=wide, deeptext=bert_model,
                        head_hidden_dims=[256, 128, 64], pred_dim=1)

    # load widedeep model
    widedeep.load_state_dict(torch.load(
        "./models/trained/wd/wd_model.pt", map_location=device))

    # load wide model
    wide.load_state_dict(torch.load(
        "./models/trained/wd/wide_model.pt", map_location=device))

    # load sbert model
    model = AutoModel.from_pretrained(
        "models/base-models/sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(
        "models/base-models/sentence-transformers/all-MiniLM-L6-v2")

    # upload to bentoml
    saved_model = bentoml.transformers.save_model("sbert", model)
    saved_tokenizer = bentoml.transformers.save_model(
        "sbert-tokenizer", tokenizer)
    saved_wd = bentoml.pytorch.save_model("widedeep", model=model)
    saved_wide = bentoml.pytorch.save_model("wide", wide)
    saved_deep = bentoml.pytorch.save_model("deep", bert_model)

    print(
        f"model saved:{saved_model},tokenizer saved:{saved_tokenizer},widedeep saved:{saved_wd}")


def upload_rec():
    rec = RecNet(9)

    rec.load_state_dict(torch.load(
        "rec_models\saved\RecNet_2023-06-29-10-27-03.pt", map_location=device))

    save_rec = bentoml.pytorch.save_model("rec", model=rec)

    print(f"model saved:{save_rec}")


if __name__ == "__main__":
    upload_rec()
