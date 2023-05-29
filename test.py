from service import svc

if __name__ == "__main__":
    for runner in svc.runners:
        runner.init_local()

    request = {
        "cve_id": "CVE-2023-0464",
        "description": "A security vulnerability has been identified in all supported versions of OpenSSL related to the verification of X.509 certificate chains that include policy constraints. Attackers may be able to exploit this vulnerability by creating a malicious certificate chain that triggers exponential use of computational resources, leading to a denial-of-service (DoS) attack on affected systems. Policy processing is disabled by default but can be enabled by passing the `-policy' argument to the command line utilities or by calling the `X509_VERIFY_PARAM_set1_policies()' function.",
        "pub_date": "2023-03-22 04:15:00",
        "cwe_type": ["CWE-295"],
        "repos": "repos/openssl",
        "files": []
    }

    result = svc.apis["rank"].func(request)

    print(result)
