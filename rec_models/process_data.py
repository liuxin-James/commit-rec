import os
import re
import json
import pandas as pd

from tqdm import tqdm
from utils.class_data import NVD
from utils.utils import extract_files, mining_commit_single, merge_featrue, mining_commit

FILE_PATH = "data_source/b_ext_vul.json"
OUT_FILE_PATH = "data_source/ext_vul.json"

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

COLS = ["text_sim","share_files_nums", "share_files_rate", "only_commit_files_nums", "exist_cve",
        "insert_loc_nums", "delete_loc_nums", "all_loc_nums", "all_method_nums", "commit_msg", "cve_desc", "commit_id", "is_right"]


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
            if len(commit_id)!=40:
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

    featrues = []

    for p, vuls in nvds.items():
        if os.path.exists(f"repos/{p}"):
            for vul in tqdm(vuls,desc=f"{p}: vuls"):
                nvd = NVD(
                    cve_id=vul["vul_id"], description=vul["description"], pub_date=vul["publish_date"], files=extract_files(vul["description"]))
                for commit_id in vul["commit_id"]:
                    commit_data = mining_commit_single(
                        repos=f"repos/{p}", commit_id=commit_id)
                    if commit_data:
                        featrue = merge_featrue(nvd, commit=commit_data)
                        featrue.append(commit_data.subject)
                        featrue.append(nvd.description)
                        featrue.append(commit_data.commit_id)
                        if commit_data.commit_id in vul["commit_id"]:
                            featrue.append(1)
                        else:
                            featrue.append(0)
                        featrues.append(featrue)
            if featrues:
                df_data = pd.DataFrame(featrues)
                df_data.to_csv("train_positive.csv", mode='a',
                            header=False, index=None)
            featrues.clear()


def build_train_dataset():
    nvds = None
    with open(OUT_FILE_PATH, "r", encoding="utf-8") as f:
        nvds = json.load(f)

    featrues = []

    for p, vuls in nvds.items():
        if os.path.exists(f"repos/{p}"):
            for vul in vuls:
                nvd = NVD(
                    cve_id=vul["vul_id"], description=vul["description"], pub_date=vul["publish_date"], files=extract_files(vul["description"]))
                for commit_id in vul["commit_id"]:
                    commit_data = mining_commit(
                        nvd=nvd, repos=f"repos/{p}")
                    if commit_data:
                        featrue = merge_featrue(nvd, commit=commit_data)
                        featrue.append(commit_data.subject)
                        featrue.append(nvd.description)
                        featrue.append(commit_data.commit_id)
                        if commit_data.commit_id in vul["commit_id"]:
                            featrue.append(1)
                        else:
                            featrue.append(0)
                        featrues.append(featrue)
        if featrues:
            df_data = pd.DataFrame(featrues)
            df_data.to_csv("train.csv", mode='a',
                           header=False, index=None)
        featrues.clear()


if __name__ == "__main__":
    extract_commit_id()
