import datetime


from utils.service_data import RequestData,Commit
from pydriller import Repository

time_delta = 3

def gen_input_data(request:RequestData):
    commits_info = get_commits_info(request.pub_date,request.repos)
    featrues = []
    for commit_info in commits_info:
        featrue = merge_featrue(request=request,commit=commit_info)
        featrue.append(commit_info.subject)
        featrue.append(request.description)
        featrue.append(commit_info.commit_id)
        featrues.append(featrue)
    return featrues

def merge_featrue(request:RequestData,commit:Commit)->list:
    featrues = []

    share_files = list(set(request.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = commit.a_file_nums - share_files_nums
    share_files_rate = round(
        share_files_nums / commit.a_file_nums, 2) if commit.a_file_nums > 0 else 0

    featrues = featrues + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description (featrues:1)
    if request.cve_id.lower() in commit.subject.lower():
        featrues.append(1)
    else:
        featrues.append(0)

    #  loc(line of code) (featrues:3)
    featrues = featrues + [commit.i_line_nums,
                           commit.d_line_nums, commit.a_line_nums]

    # method nums (featrues:1)
    featrues = featrues + [commit.a_method_nums]

    return featrues

def get_commits_info(pub_date:str,repos:str):
    since , to = gen_time_range(pub_date)
    commits = []
    for commit in Repository(path_to_repo=repos, since=since, to=to).traverse_commits():
        if not commit.in_main_branch:
            continue
        subject = commit.msg
        a_line_nums = commit.lines
        i_line_nums = commit.insertions
        d_line_nums = commit.deletions
        method_name = []
        changed_files = []
        a_file_nums = commit.files
        try:
            for files in commit.modified_files:
                for method in files.changed_methods:
                    method_name.append(method.name)
            changed_files.append(files.filename)
            a_method_nums = len(method_name)
        except Exception as ex:
            a_file_nums = 0
            a_method_nums = 0

        commits.append(
            Commit(commit_id=commit.hash,
                    subject=subject,
                    changed_files=changed_files,
                    a_line_nums=a_line_nums,
                    i_line_nums=i_line_nums,
                    d_line_nums=d_line_nums,
                    method_name=method_name,
                    a_file_nums=a_file_nums,
                    a_method_nums=a_method_nums))
    return commits


def gen_time_range(pub_date:str):
    pub_date = pub_date.split(" ")[0]
    since = datetime.datetime.strptime(
        pub_date, "%Y-%m-%d") - datetime.timedelta(days=time_delta)
    to = datetime.datetime.strptime(
        pub_date, "%Y-%m-%d") + datetime.timedelta(days=time_delta)
    
    return since,to