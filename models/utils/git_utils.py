import os
import re
import datetime
import subprocess

from dataclasses import dataclass
from utils.nvd_utils import NvdPage
from pydriller import Repository


@dataclass
class Commit:
    subject: str = None
    changed_files: list = None
    add_lines: list = None
    rm_lines: list = None
    f_nums: int = 0
    i_nums: int = 0
    d_nums: int = 0
    m_name: list = None
    m_nums: int = 0


class CommitUtils:
    time_delta = 100
    REPOS_PATH = "./repos"

    # gain commits id list by nvd publish date
    def get_commits(self, nvd_page: NvdPage):
        pub_date = nvd_page.pub_date.split(" ")[0]
        since = datetime.datetime.strptime(
            pub_date, "%Y-%m-%d") - datetime.timedelta(days=self.time_delta)
        to = datetime.datetime.strptime(
            pub_date, "%Y-%m-%d") + datetime.timedelta(days=self.time_delta)

        since = str(since).split(" ")[0]
        to = str(to).split(" ")[0]
        repos = nvd_page.repos

        cmd = f"git log  --pretty=format:'%H' --since={since} --until={to}"
        commits = self.__excute_git_cmd(self.REPOS_PATH, repos, cmd)
        commits = commits.split("\n")

        return list(set(commits))

    # gain patch information by commit id
    def get_patch(self, repos, commit_id):
        cmd = f"git format-patch -1 --stdout {commit_id}"

        p = self.__excute_git_cmd(self.REPOS_PATH, repos, cmd)

        return p

    # excute git command
    def __excute_git_cmd(self, repos_path: str, repos: str, cmd: str):
        pwd = os.getcwd()
        os.chdir(os.path.join(repos_path, repos))
        output = subprocess.check_output(
            cmd, shell=True).decode("utf-8", errors="ignore")
        os.chdir(pwd)
        return output

    # extract vulnerability type & impact & function
    def get_commit_info(self, repos, commit_id):
        commit_info = self.mining_commit_information(repos,commit_id)[0]
        return commit_info

    # mining commit information
    def mining_commit_information(self, repos: str, commit_id):
        commit_info_list = []
        for commit in Repository(path_to_repo=repos, only_commits=[commit_id]).traverse_commits():
            subject = commit.msg
            i_nums = 0
            d_nums = 0
            m_name = []
            changed_files = []
            f_nums = len(commit.modified_files)
            for files in commit.modified_files:
                i_nums += files.added_lines
                d_nums += files.deleted_lines
                for method in files.changed_methods:
                    m_name.append(method.name)
                changed_files.append(files.filename)
            m_nums = len(m_name)
            commit_info_list.append(
                Commit(subject=subject, changed_files=changed_files, i_nums=i_nums, d_nums=d_nums, m_name=m_name, f_nums=f_nums, m_nums=m_nums))
        return commit_info_list

    # utilize CodeBert to compute patch probability

    def compute_patch_prob(self, rm_lines: list, add_lines: list):
        pass