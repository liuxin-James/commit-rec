import pandas as pd
import numpy as np

from rank_net import BertModel,BertTokenizer
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.preprocessing import WidePreprocessor


def do_train():
    pass

def do_eval():
    pass


if __name__ == "__main__":
    df = pd.read_csv()
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.income_label)

    wide_cols = []
    crossed_cols = []
    # wide data preprocess
    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df_train)

    # deep data preprocess
    bert_tokenizer = BertTokenizer()
    X_bert_tr = bert_tokenizer.fit_transform(df_train["review_text"].tolist())
    X_bert_te = bert_tokenizer.transform(df_test["review_text"].tolist())

    # build model
    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
    bert_model = BertModel(freeze_bert=True)

    model = WideDeep(wide=wide, deeptabular=bert_model,head_hidden_dims=[256, 128, 64],pred_dim=2)

    trainer = Trainer(model, objective="binary", metrics=[Accuracy])


