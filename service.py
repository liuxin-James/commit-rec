import bentoml
import torch
import pickle
import pandas as pd

from bentoml.io import JSON
from utils.service_data import RequestData
from sentence_transformers import util
from utils.service_utils import gen_input_data

sbert_ref = bentoml.models.get("sbert:latest")
tokenizer_ref = bentoml.models.get("sbert-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")

ROUTE = "api/v1/commit/"

cols = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id"]

wide_preprocess = None
with open("models/trained/wd/wide_preprocess.pkl","rb") as f:
    wide_preprocess = pickle.load(f)

class CommitRecRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cuda" if torch.cuda.is_available() else "cpu")
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.sbert = bentoml.transformers.load_model(sbert_ref)

    @bentoml.Runnable.method(batchable=False)
    def rank(self,commit_msg:str,cve_desc:str):
        commit_embed = self.sbert.encode(commit_msg)
        cve_embed = self.sbert.encode(cve_desc)
        sim = util.cos_sim(commit_embed,cve_embed)
        return sim

    

commit_rec_runner = bentoml.Runner(CommitRecRunnable,name="commit_rec_runner",models=[sbert_ref,tokenizer_ref])
wd_runner = bentoml.models.get("widedeep:latest").to_runner()

svc  = bentoml.Service("commit_rec",runners=[commit_rec_runner,wd_runner])


def __check_inputs(input:dict):
    pass

@svc.api(input=JSON(),output=JSON(),route=ROUTE+"rank")
def rank(request:dict):
    request = RequestData(**request)
    featrues = gen_input_data(request=request)

    df_data = pd.DataFrame(featrues,columns=cols)
    X_wide = wide_preprocess.fit_transform(df_data)

    s = wd_runner.run(featrues)
    print(s)
