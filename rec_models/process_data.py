import os
import re
import time
import json
import torch
import pandas as pd
import concurrent.futures

from tqdm import tqdm
from torch.utils.data import Dataset
from utils.class_data import NVD, Commit
from utils.utils import extract_files, mining_commit_single, merge_feature, mining_commit

FILE_PATH = "rec_models/data_source/b_ext_vul.json"
OUT_FILE_PATH = "rec_models/data_source/ext_vul.json"

PRJ_SAMPLES = ["FFmpeg",
               "ImageMagick",
               "microweber",
               "MISP",
               "mruby",
               "openemr",
               "openssl",
               "radare2",
               "showdoc",
               "tcpdump"]

COLS = ["text_sim", "share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id", "is_right"]

max_workers = 3


def extract_commit_id():
    nvd_information = None
    commit_id_pattern = r"\bhttps://github.com/[\w.-]+/[\w.-]+(?:/pull/\d+)?/commit[s]?/([a-zA-z0-9]+)\b"

    repos_pattern = r"\bhttps://github.com/[\w.-]+/[\w.-]+\b"

    project_name_pattern = r"\bhttps://github.com/[\w.-]+/([\w.-]+)\b"

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        nvd_information = json.load(f)

    for nvd in nvd_information:
        refs = nvd.pop("reference")

        commit_id = [match.group(1)
                     for match in re.finditer(commit_id_pattern, refs)]
        nvd["commit_id"] = list(set(commit_id))

        repos_url = [match.group()
                     for match in re.finditer(repos_pattern, refs)]
        nvd["repos_url"] = list(set(repos_url))

        project_name = [match.group(1)
                        for match in re.finditer(project_name_pattern, refs)]
        nvd["project_name"] = list(set(project_name))

    nvd_information = [nvd for nvd in nvd_information if len(
        nvd["project_name"]) == 1]

    for nvd in nvd_information:
        removed_list = []
        for commit_id in nvd["commit_id"]:
            if len(commit_id) != 40:
                removed_list.append(commit_id)
        for rm in removed_list:
            nvd["commit_id"].remove(rm)
    nvd_by_project = {}

    for nvd in nvd_information:
        project_name_list = nvd.pop("project_name")
        project_name = project_name_list[0]
        if project_name in nvd_by_project.keys():
            nvd_by_project[project_name].append(nvd)
            continue
        nvd_by_project[project_name] = [nvd]

    with open(OUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(nvd_by_project))
    print("done!")


def build_positive_dataset():
    nvds = None
    with open(OUT_FILE_PATH, "r", encoding="utf-8") as f:
        nvds = json.load(f)

    features = []
    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        to_do = []
        for p, vuls in nvds.items():
            if os.path.exists(f"repos/{p}"):
                future = executor.submit(do_mining, p, vuls, True)
                to_do.append(future)
        for future in concurrent.futures.as_completed(to_do):
            features = future.result()
            if features:
                df_data = pd.DataFrame(features)
                df_data.to_csv("train_positive.csv", mode='a',
                               header=False, index=None)
    end_time = time.perf_counter()
    print(f"mining {len(nvds)} projects in {end_time-start_time} seconds")


def build_train_dataset():
    nvds = None
    with open(OUT_FILE_PATH, "r", encoding="utf-8") as f:
        nvds = json.load(f)

    featrues = []

    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        to_do = []
        for p, vuls in nvds.items():
            if os.path.exists(f"repos/{p}"):
                future = executor.submit(do_mining, p, vuls, False)
                to_do.append(future)
        for future in concurrent.futures.as_completed(to_do):
            features = future.result()
            if features:
                df_data = pd.DataFrame(features)
                df_data.to_csv("train_all.csv", mode='a',
                               header=False, index=None)
    end_time = time.perf_counter()
    print(f"mining {len(nvds)} projects in {end_time-start_time} seconds")


def build_dataset(max_workers, out_path, is_positive=True):
    nvds = None
    with open(OUT_FILE_PATH, "r", encoding="utf-8") as f:
        nvds = json.load(f)

    features = []

    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        to_do = []
        for p, vuls in nvds.items():
            if os.path.exists(f"repos/{p}"):
                future = executor.submit(do_mining, p, vuls, is_positive)
                to_do.append(future)
        for future in concurrent.futures.as_completed(to_do):
            features = future.result()
            if features:
                df_data = pd.DataFrame(features)
                df_data.to_csv(out_path, mode='a',
                               header=False, index=None)
            features.clear()
    end_time = time.perf_counter()
    print(f"mining {len(nvds)} projects in {end_time-start_time} seconds")


def do_mining(proj: str, vuls: list[dict], is_single=True):
    featrues = []
    for vul in tqdm(vuls, desc=f"{proj}: vuls", ncols=100):
        nvd = NVD(cve_id=vul["vul_id"], description=vul["description"],
                  pub_date=vul["publish_date"], files=extract_files(vul["description"]))
        for commit_id in vul["commit_id"]:
            commit_data = None
            if is_single:
                commit_data = mining_commit_single(
                    repos=f"repos/{proj}", commit_id=commit_id)
                if commit_data:
                    featrue = build_features(nvd, commit_data, vul)
                    featrues.append(featrue)
            else:
                commit_data = mining_commit(
                    nvd=nvd, repos=f"repos/{proj}")
                for commit in commit_data:
                    featrue = build_features(nvd, commit, vul)
                    featrues.append(featrue)
    return featrues


def build_features(nvd: NVD, commit: Commit, vul: dict):
    featrue = merge_feature(nvd, commit=commit)
    featrue.append(commit.subject)
    featrue.append(nvd.description)
    featrue.append(commit.commit_id)
    if commit.commit_id in vul["commit_id"]:
        featrue.append(1)
    else:
        featrue.append(0)
    return featrue


class CommitDataset(Dataset):
    def __init__(self, x_features, y_target) -> None:
        super(CommitDataset, self).__init__()
        self.x_features = x_features
        self.y_target = y_target

    def __len__(self):
        return len(self.x_features)

    def __getitem__(self, index):
        return torch.FloatTensor(self.x_features[index]), torch.LongTensor([self.y_target[index]])


if __name__ == "__main__":
    build_dataset(max_workers=5, out_path="train_dataset_v2.csv", is_positive=False)
