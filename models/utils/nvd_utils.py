import re
class NvdPage:
    def __init__(self,cve_id:str,description:str,pub_date=None,repos=None) -> None:
        self.cve_di = cve_id
        self.description = description.lower()
        self.pub_date=pub_date
        self.repos = repos
        self.files = self._extract_files()
        
    
    def _extract_files(self):
        pattern = r"[a-zA-Z_0-9]+\.[a-zA-Z]+"
        files = re.findall(pattern,self.description)
        return files