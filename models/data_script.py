import re
import json

file_path = "models/data_source/ext_vul.json"


def extract_commit_id():
    nvd_information = None
    commit_id_pattern = r"\bhttps://github.com/[\w.-]+/[\w.-]+(?:/pull/\d+)?/commit[s]?/([a-zA-z0-9]+)\b"

    repos_pattern = r"\bhttps://github.com/[\w.-]+/[\w.-]+\b"

    project_name_pattern = r"\bhttps://github.com/[\w.-]+/([\w.-]+)\b"

    with open(file_path, 'r', encoding='utf-8') as f:
        nvd_information = json.load(f)

    for nvd in nvd_information:
        refs = nvd.pop("reference")

        commit_id = [match.group(1)
                     for match in re.finditer(commit_id_pattern, refs)]
        nvd["commit_id"] = list(set(commit_id))

        repos_url = [match.group()
                     for match in re.finditer(repos_pattern, refs)]
        nvd["repos_url"] = list(set(repos_url))

        project_name = [match.group(1)
                        for match in re.finditer(project_name_pattern, refs)]
        nvd["project_name"] = list(set(project_name))

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(nvd_information))
    print("done!")


if __name__ == "__main__":
    extract_commit_id()
