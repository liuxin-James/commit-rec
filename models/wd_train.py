import torch
import pandas as pd
import numpy as np

from rank_net import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.preprocessing import WidePreprocessor
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)

def do_train():
    pass


def do_eval():
    pass


if __name__ == "__main__":
    # data part
    path = "./trainx.csv"
    df = pd.read_csv(path,encoding='gbk')
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df.is_right)

    wide_cols = ["share_files_nums", "share_files_rate", "only_commit_files_nums",
                 "exist_cve", "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums"]
    crossed_cols = [("delete_loc_nums", "all_loc_nums"), ("insert_loc_nums",
                                                          "all_loc_nums"), ("share_files_nums", "share_files_rate")]

    target = "is_right"
    target = df_train[target].values

    # wide data preprocess
    wide_preprocessor = WidePreprocessor(
        wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df_train)

    # deep training data preprocess
    bert_tokenizer = BertTokenizer()
    X_bert_tr = bert_tokenizer.fit_transform(df_train["cve_desc"].tolist())

    # model part-> build model
    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=2)
    bert_model = BertModel(freeze_bert=True)

    model = WideDeep(wide=wide, deeptext=bert_model,
                     head_hidden_dims=[256, 128, 64], pred_dim=2)

    trainer = Trainer(model, objective="binary", metrics=[Precision])

    trainer.fit(
        X_wide=X_wide,
        X_text=X_bert_tr,
        target=target,
        n_epochs=5,
        batch_size=4,
    )

    # deep testing data preprocess
    X_wide_te = wide_preprocessor.transform(df_test)
    X_bert_te = bert_tokenizer.transform(df_test["cve_desc"].tolist())

    preds = trainer.predict_proba(X_wide=X_wide_te,X_text=X_bert_te)
    pred_text_class = np.argmax(preds, 1)

    acc_score = accuracy_score(df_test.is_right, pred_text_class)
    f1_value = f1_score(df_test.is_right, pred_text_class, average="weighted")
    prec_score = precision_score(df_test.is_right, pred_text_class, average="weighted")
    rec_score = recall_score(df_test.is_right, pred_text_class, average="weighted")
    con_matrix = confusion_matrix(df_test.is_right, pred_text_class)

    print(f'accuracy_score:{acc_score},f1:{f1_value},precision_score:{prec_score},recall_score:{rec_score}')
    print(f"confusion_matrix:{con_matrix}")

    torch.save(model.state_dict(),"models/model_weights/wd_model.pt")
