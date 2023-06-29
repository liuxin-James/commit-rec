import re
import datetime

from pydriller import Repository
from nltk.corpus import stopwords
from rec_models.utils.class_data import Commit, NVD
from nltk import sent_tokenize, word_tokenize
from rec_models.utils.common_utils import init_logger
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "models/base-models/sentence-transformers/all-MiniLM-L6-v2")

TIME_DELTA = 1

logger = init_logger(__name__)


def extract_files(description: str):
    pattern = r"[a-zA-Z_0-9]+\.[a-zA-Z]+"
    files = re.findall(pattern, description)
    return files


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


def merge_feature(nvd: NVD, commit: Commit):
    features = []

    # text similarity (featrues:1)
    text_sim = 0.0
    if nvd.description and commit.subject:
        text_sim = compute_text_similarity(
            nvd.description, commit.subject).max().item()
    features = features + [text_sim]

    share_files = list(set(nvd.files) & set(commit.changed_files))
    share_files_nums = len(share_files)
    only_commit_files_nums = commit.a_file_nums - share_files_nums
    share_files_rate = round(
        share_files_nums / commit.a_file_nums, 2) if commit.a_file_nums > 0 else 0

    features = features + [share_files_nums,
                           share_files_rate, only_commit_files_nums]

    # whether contain cve_id in commit description (featrues:1)
    if nvd.cve_id.lower() in commit.subject.lower():
        features.append(1)
    else:
        features.append(0)

    #  loc(line of code) (featrues:3)
    features = features + [commit.i_line_nums,
                           commit.d_line_nums, commit.a_line_nums]

    # method nums (featrues:1)
    features = features + [commit.a_method_nums]

    return features


def mining_commit_single(repos: str, commit_id: str):
    res = None
    try:
        for commit in Repository(path_to_repo=repos, single=commit_id).traverse_commits():
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

            res = Commit(commit_id=commit.hash,
                         subject=commit.msg,
                         changed_files=changed_files,
                         a_line_nums=commit.lines,
                         i_line_nums=commit.insertions,
                         d_line_nums=commit.deletions,
                         method_name=method_name,
                         a_file_nums=a_file_nums,
                         a_method_nums=a_method_nums)
    except Exception as ex:
        pass
        # logger.info(f"mining commit EXCEPTION...{repos}-{commit_id}")
    return res


def mining_commit(nvd: NVD, repos: str) -> list[Commit]:
    pub_date = nvd.pub_date.split(" ")[0]
    since = datetime.datetime.strptime(
        pub_date, "%Y-%m-%d") - datetime.timedelta(days=TIME_DELTA)
    to = datetime.datetime.strptime(
        pub_date, "%Y-%m-%d") + datetime.timedelta(days=TIME_DELTA)

    commits = []
    try:
        for commit in Repository(path_to_repo=repos, since=since, to=to, num_workers=5).traverse_commits():
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
                       subject=commit.msg,
                       changed_files=changed_files,
                       a_line_nums=commit.lines,
                       i_line_nums=commit.insertions,
                       d_line_nums=commit.deletions,
                       method_name=method_name,
                       a_file_nums=a_file_nums,
                       a_method_nums=a_method_nums))
    except Exception as ex:
        logger.info(f"mining commit exception...{nvd.cve_id}-{repos}")
    return commits
