import bentoml
import torch

sbert_ref = bentoml.models.get("commit-rec:latest")
tokenizer_ref = bentoml.models.get("commit-rec-tokenizer:latest")
wd_ref = bentoml.models.get("widedeep:latest")


class CommitRecRunnable(bentoml.Runnable):

    def __init__(self):
        self.sbert = bentoml.transformers.load_model(sbert_ref)
        self.tokenizer = bentoml.transformers.load_model(tokenizer_ref)

    @bentoml.Runnable.method(batchable=False)
    def rank(self,commit_msg:str,cve_desc:str):
        pass


commit_rec_runner = bentoml.Runner(CommitRecRunnable,name="commit_rec_runner",models=[sbert_ref,tokenizer_ref])

svc  = bentoml.Service("commit_rec",runners=[commit_rec_runner])
