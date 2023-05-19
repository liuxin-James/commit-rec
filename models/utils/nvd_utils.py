import re

from dataclasses import dataclass

@dataclass
class NVD:
    cve_id: str = None
    description: str = None
    pub_date: str = None
    cwe_type: str = None
    repos: str = None
    files: list = None


class NvdUtils:

    # gain nvd information by cve id
    def gain_nvd_information(self, cve_id: str, repos: str=None) -> NVD:
        pass

    # extract files information by the description of NVD
    def extract_files(self, description: str):
        pattern = r"[a-zA-Z_0-9]+\.[a-zA-Z]+"
        files = re.findall(pattern, description)
        return files