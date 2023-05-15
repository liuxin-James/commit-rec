from utils.git_utils import CommitUtils
from utils.nvd_utils import NvdPage



nvd_page = NvdPage(cve_id="CVE-2021-46313",description="Hi I'm a example",pub_date="2023-04-12",repos="tqdm")

commit_utils = CommitUtils()

commits_ = commit_utils.mining_commit_information("./repos/tqdm","8fb3d91f561e2a286a7fda13291eda16613dac39")
commit = commit_utils.get_commit_info("./repos/tqdm","8fb3d91f561e2a286a7fda13291eda16613dac39")

print("this is test")