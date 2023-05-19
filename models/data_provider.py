from utils.git_utils import CommitUtils, Commit
from utils.nvd_utils import NvdUtils, NVD
from utils.text_utils import compute_text_similarity
commit_utils = CommitUtils()
nvd_utils = NvdUtils()


def merge_featrue(nvd: NVD, commit: Commit):

    featrues = []

    # text similarity (featrues:1)
    text_sim = compute_text_similarity(nvd.description, commit.subject)

    # handle files (featrues:3)
    share_files = list(set(nvd.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = len(
        list(set(commit.changed_files))) - share_files_nums
    share_files_rate = round(
        share_files_nums / len(list(set(commit.changed_files))), 2)

    featrues = featrues + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description (featrues:1)
    if nvd.cve_id in commit.subject.lower():
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

def gen_dataset(cve_info:dict[str,str], rec_commit: list) -> list:
    datasets = []
    for cve_id,repos_path in cve_info:
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