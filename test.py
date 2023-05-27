from service import svc

for runner in svc.runners:
    runner.init_local()

request = {
    "cve_id": "CVE-2021-31780",
    "description": "In app/Model/MispObject.php in MISP 2.4.141, an incorrect sharing group association could lead to information disclosure on an event edit. When an object has a sharing group associated with an event edit, the sharing group object is ignored and instead the passed local ID is reused.",
    "pub_date": "2021-04-24 04:15:00",
    "cwe_type": ["CWE-212"],
    "repos": "https://github.com/MISP/MISP",
}

result = svc.apis["rank"].func(request)

print(result)
