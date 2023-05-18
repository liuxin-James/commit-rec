from utils.git_utils import CommitUtils, Commit
from utils.nvd_utils import NvdUtils, NVD
from utils.text_utils import compute_text_similarity
commit_utils = CommitUtils()
nvd_utils = NvdUtils()


def merge_featrue(nvd: NVD, commit: Commit):

    featrues = []

    # text similarity
    text_sim = compute_text_similarity(nvd.description, commit.subject)

    # handle files
    share_files = list(set(nvd.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = len(
        list(set(commit.changed_files))) - share_files_nums
    share_files_rate = round(
        share_files_nums / len(list(set(commit.changed_files))), 2)
    
    featrues = featrues + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description
    if nvd.cve_id in commit.subject.lower():
        featrues.append(1)
    else:
        featrues.append(0)

    # 
    pass


def gain_featrue(cve_id: str, repos: str):
    nvd = nvd_utils.gain_nvd_information(cve_id=cve_id, repos=repos)
    nvd.files = nvd_utils._extract_files(nvd.description)

    commits = commit_utils.get_commits(nvd_page=nvd)

    for commit_id in commits:
        commit = commit_utils.get_commit_info(
            repos=nvd.repos, commit_id=commit_id)
