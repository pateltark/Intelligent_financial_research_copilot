import requests, os
from dotenv import load_dotenv

load_dotenv()

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}

def get_latest_filing(ticker_cik: str, form_type: str = "10-K"):
    # CIK zero-pad to 10 digits
    cik = ticker_cik.zfill(10)
    
    # Get submissions
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()
    
    filings = data["filings"]["recent"]
    
    # Find first match
    for i, form in enumerate(filings["form"]):
        if form == form_type:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name  = filings["primaryDocument"][i]
            cik_raw   = cik.lstrip("0")
            
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accession}/{doc_name}"
            return doc_url, doc_name
    
    return None, None

def download_filing(cik: str, form_type: str = "10-K", save_dir: str = "edgar_docs"):
    os.makedirs(save_dir, exist_ok=True)
    
    doc_url, doc_name = get_latest_filing(cik, form_type)
    if not doc_url:
        print(f"No {form_type} found.")
        return
    
    print(f"Fetching: {doc_url}")
    r = requests.get(doc_url, headers=HEADERS)
    
    filepath = os.path.join(save_dir, doc_name)
    with open(filepath, "wb") as f:
        f.write(r.content)
    
    print(f"Saved: {filepath}")
    return filepath

# Usage — CIK for Tesla
download_filing("1318605", form_type="10-K")