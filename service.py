import bentoml
import torch

from bentoml.io import JSON
from utils.service_data import RequestData

sbert_ref = bentoml.models.get("commit-rec:latest")
tokenizer_ref = bentoml.models.get("commit-rec-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")

ROUTE = "api/v1/rec/"
class CommitRecRunnable(bentoml.Runnable):

    def __init__(self):
        self.sbert = bentoml.transformers.load_model(sbert_ref)
        self.tokenizer = bentoml.transformers.load_model(tokenizer_ref)

    @bentoml.Runnable.method(batchable=False)
    def rank(self,commit_msg:str,cve_desc:str):
        pass


commit_rec_runner = bentoml.Runner(CommitRecRunnable,name="commit_rec_runner",models=[sbert_ref,tokenizer_ref])

svc  = bentoml.Service("commit_rec",runners=[commit_rec_runner])



def __check_inputs(input:dict):
    pass

@svc.api(input=JSON(),output=JSON(),route=ROUTE+"commit")
def rank(request:RequestData):
    pass