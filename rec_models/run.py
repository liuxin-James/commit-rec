import pandas as pd
import torch

from utils.common_utils import init_logger
from models import RecNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from process_data import CommitDataset
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

COLS = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id", "is_right"]

X_COLS = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
          "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums"]
Y_COLS = ["is_right"]

model = SentenceTransformer(
    "models/base-models/sentence-transformers/all-MiniLM-L6-v2")

config = {
    "lr": 1e-5,
    "epochs": 300,
    "seed": 42,
    "batch_size": 4,
    "ratio": 0.2,
    "eval_step": 4185
}

logger = init_logger(__name__)


def train(model: torch.nn.Module, train_dataloader: DataLoader, config: dict, eval_dataloader: DataLoader = None):
    global_step = 0
    tr_loss = 0.0

    logger.info("\n")
    logger.info(
        "*****************Training Parameters Information*****************")
    logger.info(f"Current learning rate:{config['lr']}")
    logger.info(
        f"Training datasize:{len(train_dataloader)*config['batch_size']}")
    logger.info(f"Training epochs:{config['epochs']}")
    logger.info(f"Training batch size:{config['batch_size']}")
    logger.info(
        "*****************Training Fun*****************")
    model.train()

    epochs = config["epochs"]

    progress_bar = tqdm(range(len(train_dataloader)), ncols=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(epochs)):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}\n--------------")
        progress_bar.reset()
        progress_bar.set_description(f"loss value:{0:>7f}")

        n_epoch_total_loss = 0
        finished_batch_num = epoch * len(train_dataloader)

        for batch, (X, y) in enumerate(train_dataloader, start=1):
            X, y = X.to(config["device"]), y.to(config["device"])
            optimizer.zero_grad()

            loss, logits = model(X, y)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()

            n_epoch_total_loss += loss.item()

            progress_bar.set_description(
                f"loss value:{n_epoch_total_loss/(finished_batch_num+batch):>7f}")
            progress_bar.update(1)
            global_step += 1

            if config["eval_step"] > 0 and global_step % config["eval_step"] == 0 and eval_dataloader is not None:
                evaluate(model=model, eval_dataloader=eval_dataloader,
                         config=config)
                model.train()

        logger.info("\n")
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}: loss value:{tr_loss/global_step}")


def evaluate(model: torch.nn.Module, eval_dataloader: DataLoader, config: dict):
    klasses, predictions = [], []
    model.eval()

    logger.info("start evaluate model...")

    with torch.no_grad():
        progress_bar = tqdm(range(len(eval_dataloader)), ncols=100)
        for X, y in eval_dataloader:
            X, y = X.to(config["device"]), y.to(config["device"])

            pred = model(X)
            pred_ = pred[0].argmax(dim=-1).cpu().numpy().tolist()

            y_ = y.squeeze(1).cpu().numpy().tolist()

            predictions += pred_
            klasses += y_

            progress_bar.update(1)

        logger.info(classification_report(klasses, predictions))
        f1_value = f1_score(klasses, predictions)
        precision_value = precision_score(klasses, predictions)
        recall_value = recall_score(klasses, predictions)
        accuracy_value = accuracy_score(klasses, predictions)
        logger.info(
            f"f1 value:{f1_value},\nprecision value:{precision_value},\nrecall value:{recall_value},\naccuracy value:{accuracy_value}\n")


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
    x, y = train_dataset[3]
    # build Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # build rec model
    model = RecNet(8)
    model.to(config["device"])

    # model training
    train(model=model, train_dataloader=train_dataloader,
          config=config, eval_dataloader=test_dataloader)


if __name__ == "__main__":
    main()
