import bentoml
import torch
import pickle
import pandas as pd

from bentoml.io import JSON
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Precision
from pytorch_widedeep.models import WideDeep
from utils.service_data import RequestData
from sentence_transformers import util
from utils.service_utils import gen_input_data

# get model ref from bentoml
sbert_ref = bentoml.models.get("sbert:latest")
tokenizer_ref = bentoml.models.get("sbert-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")
wide_ref = bentoml.models.get("wide:latest")
deep_ref = bentoml.models.get("deep:latest")

# define api route
ROUTE = "api/v1/commit/"

# DataFrame cols name
cols = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id"]

# load wide data preprocess
wide_preprocess = None
with open("models/trained/wd/wide_preprocess.pkl", "rb") as f:
    wide_preprocess = pickle.load(f)

# define custom commit rec runner


class CommitRecRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cuda" if torch.cuda.is_available() else "cpu")
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.sbert = bentoml.transformers.load_model(sbert_ref)
        self.wide = bentoml.pytorch.load_model(wide_ref)
        self.deep = bentoml.pytorch.load_model(deep_ref)
        self.widedeep = WideDeep(wide=self.wide, deeptext=self.deep, head_hidden_dims=[
                                 256, 128, 64], pred_dim=1)

    @bentoml.Runnable.method(batchable=False)
    def compute_text_sim(self, commit_msg: str, cve_desc: str):
        commit_embed = self.sbert.encode(commit_msg)
        cve_embed = self.sbert.encode(cve_desc)
        sim = util.cos_sim(commit_embed, cve_embed)
        return sim

    @bentoml.Runnable.method(batchable=False)
    def widedeep_do(self, request: RequestData):
        featrues = gen_input_data(request=request)
        df_data = pd.DataFrame(featrues, columns=cols)
        X_wide = wide_preprocess.fit_transform(df_data)
        trainer = Trainer(model=self.widedeep,
                          objective="binary", metrics=[Precision])
        cve_embed = self.sbert.encode(df_data["cve_desc"].tolist())

        preds = trainer.predict_proba(X_wide=X_wide, X_text=cve_embed)

        return preds


# load custom commit rec runner
commit_rec_runner = bentoml.Runner(
    CommitRecRunnable, name="commit_rec_runner", models=[sbert_ref, tokenizer_ref, wd_ref])

svc = bentoml.Service("commit_rec", runners=[commit_rec_runner])

# check request parameters TODO


def __check_inputs(input: dict):
    pass


@svc.api(input=JSON(), output=JSON(), route=ROUTE+"rank")
def rank(request: dict):
    request = RequestData(**request)
    proba = commit_rec_runner.widedeep_do.run(request)
    print(proba)
