import bentoml
import torch

from bentoml.io import JSON
from utils.service_data import RequestData
from sentence_transformers import util
from utils.service_utils import gen_input_data

sbert_ref = bentoml.models.get("sbert:latest")
tokenizer_ref = bentoml.models.get("sbert-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")

ROUTE = "api/v1/rec/"
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

@svc.api(input=JSON(),output=JSON(),route=ROUTE+"commit")
def rank(request:RequestData):
    featrues = gen_input_data(request=request)
    s = wd_runner.run(featrues)
    print(s)
