import bentoml
import torch
import pickle
import numpy as np
import pandas as pd

from bentoml.io import JSON
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Precision
from pytorch_widedeep.models import WideDeep
from utils.service_data import RequestData
from sentence_transformers import util
from utils.service_utils import gen_input_data
from nltk import sent_tokenize
from models.rank_net import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from models.utils.text_utils import compute_text_similarity
from utils.service_data import ResponseCode, Response
from rec_models.process_data import CommitDataset
from torch.utils.data import DataLoader

# get model ref from bentoml
sbert_ref = bentoml.models.get("sbert:latest")
tokenizer_ref = bentoml.models.get("sbert-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")
wide_ref = bentoml.models.get("wide:latest")
deep_ref = bentoml.models.get("deep:latest")
rec_ref = bentoml.models.get("rec:latest")
sentence_bert = SentenceTransformer(
    "models/base-models/sentence-transformers/all-MiniLM-L6-v2")
tokenizer_z = BertTokenizer()
# define api route
ROUTE = "api/v1/commit/"

# DataFrame cols name
cols = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id"]

# load wide data preprocess
wide_preprocess = None
with open("models/trained/wd/wide_preprocess.pkl", "rb") as f:
    wide_preprocess = pickle.load(f)


prob_threshold = 0.4
sim_threshold = 0.25
rough_n_top = 10
fine_n_top = 5

# define custom commit rec runner


class CommitRecRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cuda" if torch.cuda.is_available() else "cpu")
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.sbert = bentoml.transformers.load_model(sbert_ref)
        self.wide = bentoml.pytorch.load_model(wide_ref)
        self.deep = BertModel(freeze_bert=True)
        self.tokenizer = bentoml.transformers.load_model(tokenizer_ref)
        self.widedeep = WideDeep(wide=self.wide, deeptext=self.deep, head_hidden_dims=[
                                 256, 128, 64], pred_dim=1)

    def compute_text_sim(self, commit_msg: str, cve_desc: str):
        commit_embed = self.sbert.encode(commit_msg)
        cve_embed = self.sbert.encode(cve_desc)
        sim = util.cos_sim(commit_embed, cve_embed)
        return sim

    def rough_sort(self, df_data: pd.DataFrame):
        sent_sim = []
        for index, row in df_data.iterrows():
            sent_sim.append(compute_text_similarity(
                row["commit_msg"], row["cve_desc"]).data.cpu().mean().numpy())
        df_data["text_sim"] = sent_sim
        df_data = df_data.sort_values(by="text_sim", ascending=False)
        return df_data

    def fine_sort(self, X_wide, X_text, y_df):
        trainer = Trainer(self.widedeep, objective="binary",
                          metrics=[Precision])
        preds = trainer.predict_proba(X_wide=X_wide, X_text=X_text)
        pred_commit_class = np.argmax(preds, 1).reshape(-1, 1)
        pred_prob = np.max(preds, axis=1).reshape(-1, 1)
        y_df["klass"] = pred_commit_class
        y_df["prob"] = pred_prob
        return y_df

    @bentoml.Runnable.method(batchable=False)
    def rec(self, request: RequestData):
        featrues = gen_input_data(request=request)
        df_data = pd.DataFrame(featrues, columns=cols)

        # compute text similarity & filter
        df_data = self.rough_sort(df_data=df_data)
        df_data.drop(df_data[df_data.text_sim <
                     sim_threshold].index, inplace=True)

        X_wide = wide_preprocess.transform(df_data)

        res_df = df_data[["commit_id", "commit_msg"]]

        X_text = tokenizer_z.fit(df_data["cve_desc"].tolist()).transform(
            df_data["cve_desc"].tolist())

        res_df = self.fine_sort(X_wide=X_wide, X_text=X_text, y_df=res_df)

        res_df.drop(res_df[res_df.klass == 0].index, inplace=True)
        res_df.sort_values(by="prob", ascending=False)

        fine_sort_top_n = res_df.iloc[:fine_n_top, :]

        results = []
        for index, row in fine_sort_top_n.iterrows():
            results.append(
                {"commit_id": row["commit_id"], "prob": row["prob"], "klass": row["klass"]})
        return results


class MLPRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cuda" if torch.cuda.is_available() else "cpu")
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.model = bentoml.pytorch.load_model(rec_ref)
        self.cols = ["text_sim", "share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
                     "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id"]
        self.x_cols = ["text_sim", "share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
                       "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums"]

    def classify(self, dataloader: DataLoader, y_df: pd.DataFrame):
        self.model.eval()
        preds, probs = [], []

        for X in dataloader:
            X = X.to(self.SUPPORTED_RESOURCES)
            pred = self.model(X)
            pred_ = pred[0].argmax(dim=-1).cpu().numpy().tolist()
            prob = pred[0].max(dim=-1).values.detach().cpu().numpy().tolist()
            preds += pred_
            probs += prob
        y_df["klass"] = preds
        y_df["prob"] = probs
        return y_df

    @bentoml.Runnable.method(batchable=False)
    def rec(self, request: RequestData):
        featrues = gen_input_data(request=request)
        df_data = pd.DataFrame(featrues, columns=self.cols)

        res_df = df_data[["commit_id", "commit_msg"]]
        x_df_data = df_data[self.x_cols]
        # prepare dataset
        data = x_df_data.values
        dataset = CommitDataset(x_features=data)
        dataloader = DataLoader(dataset=dataset, batch_size=4)

        # compute text similarity & filter
        res_df = self.classify(dataloader=dataloader, y_df=res_df)

        res_df.drop(res_df[res_df.klass == 0].index, inplace=True)
        res_df.sort_values(by="prob", ascending=False)

        fine_sort_top_n = res_df.iloc[:fine_n_top, :]

        results = []
        for index, row in fine_sort_top_n.iterrows():
            results.append(
                {"commit_id": row["commit_id"], "prob": row["prob"], "klass": row["klass"]})
        return results


# load custom commit rec runner
# commit_rec_runner = bentoml.Runner(
#     CommitRecRunnable, name="commit_rec_runner", models=[sbert_ref, tokenizer_ref, wide_ref, deep_ref, wd_ref])

# svc = bentoml.Service("commit_rec", runners=[commit_rec_runner])

mlp_rec_runner = bentoml.Runner(
    MLPRunnable, name="mlp_rec_runner", models=[rec_ref])

svc = bentoml.Service("mlp_rec", runners=[mlp_rec_runner])

# check request parameters TODO


def __check_inputs(request: RequestData):
    if not request.cve_id:
        raise ValueError("cve_id is invalid!please check...")
    if not request.repos:
        raise ValueError("repos is invalid!please check...")


@svc.api(input=JSON(), output=JSON(), route=ROUTE+"rank")
def rank(request: dict):
    request = RequestData(**request)
    response = Response()
    res_df = None

    try:
        __check_inputs(request=request)
        res_df = mlp_rec_runner.rec.run(request)
        response.status = ResponseCode.Success.value
        response.msg = "Success"
        response.result = res_df
    except Exception as ex:
        response.status = ResponseCode.Exception.value
        response.msg = str(ex)
    return response.__dict__
