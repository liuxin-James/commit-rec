from dataclasses import dataclass

@dataclass
class NVD:
    cve_id: str = None
    description: str = None
    pub_date: str = None
    cwe_type: str = None
    repos: str = None
    files: list = None

@dataclass
class Commit:
    commit_id: str = None
    subject: str = ""
    changed_files: list = None
    add_lines: list = None
    rm_lines: list = None
    a_file_nums: int = 0
    i_line_nums: int = 0
    d_line_nums: int = 0
    a_line_nums: int = 0
    method_name: list = None
    a_method_nums: int = 0