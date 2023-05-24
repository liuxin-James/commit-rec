import re
import json
import pandas as pd

from tqdm import tqdm
from utils.git_utils import CommitUtils, Commit
from utils.nvd_utils import NvdUtils, NVD
from utils.text_utils import compute_text_similarity
commit_utils = CommitUtils()
nvd_utils = NvdUtils()

data_source_path = "models/data_source/ext_vul.json"

project_samples = ["FFmpeg",
                   "ImageMagick",
                   "microweber",
                   "MISP",
                   "mruby",
                   "openemr",
                   "openssl",
                   "radare2",
                   "showdoc",
                   "tcpdump"]

cols = ["share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id", "is_right"]


def merge_featrue(nvd: NVD, commit: Commit):

    featrues = []

    # text similarity (featrues:1)
    # text_sim = compute_text_similarity(nvd.description, commit.subject)

    # handle files (featrues:3)
    share_files = list(set(nvd.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = commit.a_file_nums - share_files_nums
    share_files_rate = round(
        share_files_nums / commit.a_file_nums, 2) if commit.a_file_nums > 0 else 0

    featrues = featrues + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description (featrues:1)
    if nvd.cve_id.lower() in commit.subject.lower():
        featrues.append(1)
    else:
        featrues.append(0)

    #  loc(line of code) (featrues:3)
    featrues = featrues + [commit.i_line_nums,
                           commit.d_line_nums, commit.a_line_nums]

    # method nums (featrues:1)
    featrues = featrues + [commit.a_method_nums]

    return featrues


def gain_featrue(cve_id: str, repos_path: str):
    nvd = nvd_utils.gain_nvd_information(cve_id=cve_id)
    nvd.files = nvd_utils._extract_files(nvd.description)

    commits = commit_utils.get_commits(nvd_page=nvd, repos_path=repos_path)

    featrues = []
    for commit_id in commits:
        commit = commit_utils.get_commit_info(
            repos=repos_path, commit_id=commit_id)
        featrue = merge_featrue(nvd=nvd, commit=commit)
        featrues.append(featrue)
    return featrues


def gen_dataset(cve_id: str, repos_path: str, rec_commit: list) -> list:
    nvd = nvd_utils.gain_nvd_information(cve_id=cve_id)
    commits = commit_utils.get_commits(nvd_page=nvd, repos_path=repos_path)

    featrues = []
    for commit_id in commits:
        commit = commit_utils.get_commit_info(
            repos=repos_path, commit_id=commit_id)
        featrue = merge_featrue(nvd=nvd, commit=commit)
        featrue.append(commit_id)
        if commit_id in rec_commit:
            featrue.append(1)
        else:
            featrue.append(0)
        featrues.append(featrue)
    return featrues


def gen_dataset(cve_info: dict[str, str], rec_commit: list) -> list:
    datasets = []
    for cve_id, repos_path in cve_info:
        nvd = nvd_utils.gain_nvd_information(cve_id=cve_id)
        commits = commit_utils.get_commits(nvd_page=nvd, repos_path=repos_path)

        featrues = []
        for commit_id in commits:
            commit = commit_utils.get_commit_info(
                repos=repos_path, commit_id=commit_id)
            featrue = merge_featrue(nvd=nvd, commit=commit)
            featrue.append(commit_id)
            if commit_id in rec_commit:
                featrue.append(1)
            else:
                featrue.append(0)
            featrues.append(featrue)
        datasets.append(featrues)
    return datasets


def build_train_dataset():
    nvds = None
    with open(data_source_path, 'r', encoding='utf-8') as f:
        nvds = json.load(f)
    featrues = []
    for nvd in tqdm(nvds, desc="nvd nums"):
        for p in nvd["project_name"]:
            if p in project_samples:
                n = NVD(
                    cve_id=nvd["vul_id"], description=nvd["description"], pub_date=nvd["publish_date"], files=nvd_utils.extract_files(nvd["description"]))

                commits = commit_utils.mining_commit_information(
                    nvd=n, repos_path=f"repos/{p}")
                for commit in commits:
                    featrue = merge_featrue(n, commit)
                    featrue.append(commit.subject)
                    featrue.append(n.description)
                    featrue.append(commit.commit_id)
                    if commit.commit_id in nvd["commit_id"]:
                        featrue.append(1)
                    else:
                        featrue.append(0)
                    featrues.append(featrue)
                if featrues:
                    df_data = pd.DataFrame(featrues)
                    df_data.to_csv("train.csv", mode='a',
                                   header=False, index=None)
                featrues.clear()
    print("done!")


def build_positive_dataset():
    nvds = None
    with open(data_source_path, 'r', encoding='utf-8') as f:
        nvds = json.load(f)
    featrues = []

    for nvd in tqdm(nvds, desc="nvd nums"):
        for p in nvd["project_name"]:
            if p in project_samples:
                n = NVD(
                    cve_id=nvd["vul_id"], description=nvd["description"], pub_date=nvd["publish_date"], files=nvd_utils.extract_files(nvd["description"]))
                for commit in nvd["commit_id"]:
                    cmt = commit_utils.mining_single_commit_information(
                        repos=f"repos/{p}", commit_id=commit)
                    featrue = merge_featrue(n, cmt)
                    featrue.append(cmt.subject)
                    featrue.append(n.description)
                    featrue.append(cmt.commit_id)
                    if cmt.commit_id in nvd["commit_id"]:
                        featrue.append(1)
                    else:
                        featrue.append(0)
                    featrues.append(featrue)
                if featrues:
                    df_data = pd.DataFrame(featrues)
                    df_data.to_csv("train.csv", mode='a',
                                   header=False, index=None)
                featrues.clear()
    print("done!")


if __name__ == "__main__":
    build_positive_dataset()
    # build_train_dataset()
