from service import svc

if __name__ == "__main__":
    for runner in svc.runners:
        runner.init_local()

    request = {
    "cve_id": "CVE-2022-40743",
    "description": "Improper Input Validation vulnerability for the xdebug plugin in Apache Software Foundation Apache Traffic Server can lead to cross site scripting and cache poisoning attacks.This issue affects Apache Traffic Server: 9.0.0 to 9.1.3. Users should upgrade to 9.1.4 or later versions.",
    "pub_date": "2022-12-19 00:15:00",
    "cwe_type": [
        "CWE-20"
    ],
    "repos": "repos/trafficserver",
    "files": []
}

    result = svc.apis["rank"].func(request)

    print(result)
