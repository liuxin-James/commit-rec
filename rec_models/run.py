import pandas as pd
import torch

from models import RecNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from process_data import CommitDataset
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


COLS = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id", "is_right"]

X_COLS = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
          "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums"]
Y_COLS = ["is_right"]

model = SentenceTransformer(
    "models/base-models/sentence-transformers/all-MiniLM-L6-v2")

config = {
    "lr":1e-5,
    "epochs":300,
    "seed":42,
    "batch_size":64,
    "ratio":0.2
}

def train(model: torch.nn.Module, train_dataloader: DataLoader, config: dict):
    progress_bar = tqdm(train_dataloader, position=0, leave=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    model.train()

    epochs = config["epochs"]

    for epoch  in tqdm(range(epochs)):

        for X,y in progress_bar:
            X = X.to(config["device"])
            y = y.to(config["device"])
            optimizer.zero_grad()

            loss,logits = model(X,y)
            loss.backward()
            optimizer.step()


def evaluate():
    pass


def predict():
    pass


def main():
    # read data
    x_df = pd.read_csv("train_dataset.csv", usecols=X_COLS).values
    y_df = pd.read_csv("train_dataset.csv", usecols=Y_COLS).values[:, 0]

    # resampling using SMOTE
    x_resampled, y_resampled = BorderlineSMOTE().fit_resample(X=x_df, y=y_df)

    print(sorted(Counter(y_df).items()))
    print(sorted(Counter(y_resampled).items()))

    # train & test dataset split
    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled, test_size=config["ratio"], random_state=config["seed"])

    # build dataset
    train_dataset = CommitDataset(x_features=x_train, y_target=y_train)
    test_dataset = CommitDataset(x_features=x_test, y_target=y_test)
    x,y = train_dataset[3]
    # build Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=4, shuffle=True)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # build rec model
    model = RecNet(8)

    # model training
    train(model=model,train_dataloader=train_dataloader,config=config)

if __name__ == "__main__":
    main()
