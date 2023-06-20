from enum import Enum
from dataclasses import dataclass


@dataclass
class RequestData:
    cve_id: str = None
    description: str = None
    pub_date: str = None
    cwe_type: str = None
    repos: str = None
    from_tag:str = None
    to_tag:str = None
    from_time:str = None
    to_time:str = None
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


class ResponseCode(Enum):
    Fail = 0
    Success = 1
    Exception = 4


@dataclass
class Response:
    def __init__(self, status=ResponseCode.Fail.value, result=None, msg=None):
        self.status = status
        self.result = result
        self.msg = msg
