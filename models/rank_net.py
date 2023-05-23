import torch
import numpy as np
import torch.nn as nn

from torch import Tensor, nn
from transformers import BertModel,DistilBertModel,DistilBertTokenizer
from pytorch_widedeep.utils.fastai_transforms import (
    fix_html,
    spec_add_spaces,
    rm_useless_spaces,
)

class BertTokenizer(object):
    def __init__(
        self,
        pretrained_tokenizer="models/base-models/distilbert-base-uncased",
        do_lower_case=True,
        max_length=90,
    ):
        super(BertTokenizer, self).__init__()
        self.pretrained_tokenizer = pretrained_tokenizer
        self.do_lower_case = do_lower_case
        self.max_length = max_length

    def fit(self, texts):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.pretrained_tokenizer, do_lower_case=self.do_lower_case
        )

        return self

    def transform(self, texts):
        input_ids = []
        for text in texts:
            encoded_sent = self.tokenizer.encode_plus(
                text=self._pre_rules(text),
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

            input_ids.append(encoded_sent.get("input_ids"))
        return np.stack(input_ids)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @staticmethod
    def _pre_rules(text):
        return fix_html(rm_useless_spaces(spec_add_spaces(text)))

class BertModel(nn.Module):
    def __init__(
        self,
        model_name: str = "models/base-models/distilbert-base-uncased",
        freeze_bert: bool = False,
    ):
        super(BertModel, self).__init__()

        self.bert = DistilBertModel.from_pretrained(
            model_name,
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, X_inp: Tensor) -> Tensor:
        attn_mask = (X_inp != 0).type(torch.int8)
        outputs = self.bert(input_ids=X_inp, attention_mask=attn_mask)
        return outputs[0][:, 0, :]

    @property
    def output_dim(self) -> int:
        # This is THE ONLY requirement for any model to work with pytorch-widedeep. Must
        # have a 'output_dim' property so the WideDeep class knows the incoming dims
        # from the custom model. in this case, I hardcoded it
        return 768


class DeepBert(nn.Module):
    def __ini__(self, name_or_path, freeze_bert: bool = True):
        super(DeepBert, self).__init__()
        self.bert = BertModel.from_pretrained(name_or_path)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        outputs = self.bert(**inputs)


class MLPNet(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x1, x2):
        o1 = self.model(x1)
        o2 = self.model(x2)
        return self.output(o1-o2)

    def predict(self, x):
        return self.model(x)
