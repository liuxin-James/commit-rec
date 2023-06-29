import datetime

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.service_data import RequestData,Commit
from pydriller import Repository
from sentence_transformers import SentenceTransformer, util

time_delta = 5
model = SentenceTransformer(
    "models/base-models/sentence-transformers/all-MiniLM-L6-v2")

def compute_text_similarity(sentence1: str, sentence2: str):
    sentence1 = sent_tokenize(preprocess_sentence(sentence1))
    sentence2 = sent_tokenize(preprocess_sentence(sentence2))

    embedding1 = model.encode(sentences=sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentences=sentence2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embedding1, embedding2)

    return cosine_scores

def preprocess_sentence(sentence: str):
    before_words = word_tokenize(sentence)
    after_words = [
        word for word in before_words if word not in stopwords.words("english")]
    sentence_ = " ".join(after_words)
    return sentence_

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
    features = []

    text_sim = 0.0
    if request.description and commit.subject:
        text_sim = compute_text_similarity(
            request.description, commit.subject).max().item()
    features = features + [text_sim]

    share_files = list(set(request.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = commit.a_file_nums - share_files_nums
    share_files_rate = round(
        share_files_nums / commit.a_file_nums, 2) if commit.a_file_nums > 0 else 0

    features = features + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description (featrues:1)
    if request.cve_id.lower() in commit.subject.lower():
        features.append(1)
    else:
        features.append(0)

    #  loc(line of code) (featrues:3)
    features = features + [commit.i_line_nums,
                           commit.d_line_nums, commit.a_line_nums]

    # method nums (featrues:1)
    features = features + [commit.a_method_nums]

    return features

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