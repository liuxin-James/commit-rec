from utils.git_utils import CommitUtils
from utils.nvd_utils import NvdPage



nvd_page = NvdPage(cve_id="CVE-2021-46313",description="Hi I'm a example",pub_date="2023-04-12",repos="tqdm")

commit_utils = CommitUtils()

commit_utils.get_commits(nvd_page=nvd_page)
    