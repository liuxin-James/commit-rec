import os
import datetime
import subprocess

from utils.nvd_utils import NvdPage


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
        commits = self.__excute_git_cmd(self.REPOS_PATH, repos,cmd)
        commits = commits.split("\n")

        return list(set(commits))

    # gain patch information by commit id
    def get_patch(self, repos, commit_id):
        cmd = f"git format-patch -1 --stdout {commit_id}"

        p = self.__excute_git_cmd(self.REPOS_PATH,repos,cmd)

        return p

    # excute git command
    def __excute_git_cmd(self, repos_path:str,repos:str,cmd: str):
        pwd = os.getcwd()
        os.chdir(os.path.join(repos_path, repos))
        output = subprocess.check_output(
            cmd, shell=True).decode("utf-8", errors="ignore")
        os.chdir(pwd)
        return output

    # extract vulnerability type & impact & function
    def get_commit_info(repos, commit_id):
        pass

    # utilize CodeBert to compute patch probability
    def compute_patch_prob(rm_lines: list, add_lines: list):
        pass
