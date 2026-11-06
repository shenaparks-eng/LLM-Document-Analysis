## Shena Parks/CAP5619/ Assignment 2

import csv
import json
import os
import time
from bs4 import BeautifulSoup
import ollama
import requests

# SEC EDGAR requires a custom User-Agent with your name and institutional email
HEADERS = {
    "User-Agent": "Shena Parks shena_parks@ucf.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

MODEL_NAME = "llama3.2"
TARGET_ENTRIES = 100


def get_company_cik(ticker: str) -> str | None:
    """Finds the zero-padded 10-digit CIK number for a given stock ticker."""
    url = "https://www.sec.gov/files/company_tickers.json"
    ticker_headers = HEADERS.copy()
    ticker_headers["Host"] = "www.sec.gov"

    time.sleep(0.2)
    response = requests.get(url, headers=ticker_headers)
    if response.status_code != 200:
        return None

    data = response.json()
    ticker = ticker.upper()
    for _, company in data.items():
        if company["ticker"] == ticker:
            return str(company["cik_str"]).zfill(10)
    return None


def fetch_8k_filing_links(cik: str) -> list[dict]:
    """Fetches recent Form 8-K filing details from the company's submissions JSON."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    time.sleep(0.2)
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return []

    data = response.json()
    company_name = data.get("name", "Unknown Corp")
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])

    filings = []
    for form, filing_date, acc_num, primary_doc in zip(
            forms, filing_dates, accession_numbers, primary_documents
    ):
        if form == "8-K":
            acc_no_dashes = acc_num.replace("-", "")
            cik_int = int(cik)
            link = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{primary_doc}"
            filings.append({
                "company_name": company_name,
                "filing_date": filing_date,
                "link": link,
            })

    return filings


def fetch_filing_text(filing_url: str) -> str:
    """Downloads and extracts SEC archive HTML document text."""
    doc_headers = HEADERS.copy()
    doc_headers["Host"] = "www.sec.gov"

    time.sleep(0.2)
    try:
        response = requests.get(filing_url, headers=doc_headers)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()

        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return ""


def extract_product_with_llm(
        company_name: str, stock_name: str, filing_text: str
) -> dict:
    """Uses Ollama to perform NER and extract newly announced product details."""
    prompt = f"""
    You are an expert financial and Named Entity Recognition (NER) analyst.
    Analyze the following SEC Form 8-K filing text for {company_name} ({stock_name}).

    Extract the following details:
    1. "company_name": The exact company name provided.
    2. "stock_name": The ticker symbol provided.
    3. "filing_time": The date of the filing.
    4. "new_product": The exact name of any newly announced product, service, or major release. If none, return "None".
    5. "product_description": A concise description of the product, strictly limited to less than 180 characters. If none, return "None".

    Output strictly in valid JSON format with keys: 
    "company_name", "stock_name", "filing_time", "new_product", "product_description".

    Filing text:
    {filing_text[:3000]}
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        desc = data.get("product_description", "None")
        if desc and len(desc) > 180:
            data["product_description"] = desc[:177] + "..."

        return data
    except Exception:
        return {
            "company_name": company_name,
            "stock_name": stock_name,
            "filing_time": "2026-01-01",
            "new_product": "None",
            "product_description": "None",
        }


def main():
    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", "sec_8k_products.csv")

    # Expanded basket of major tickers to ensure you hit 100+ entries easily
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "NFLX", "ADBE", "INTC", "AMD", "CRM", "ORCL", "IBM", "QCOM"
    ]

    # Keyword filter to speed up text scanning
    PRODUCT_KEYWORDS = [
        "launch", "releases", "announced", "introduces",
        "unveils", "new product", "hardware", "software"
    ]

    extracted_count = 0
    print(f"[*] Starting SEC 8-K pipeline. Target: {TARGET_ENTRIES} entries...")

    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["company_name", "stock_name", "filing_time", "new_product", "product_description"])

        for ticker in tickers:
            if extracted_count >= TARGET_ENTRIES:
                break

            print(f"[+] Fetching CIK for ticker: {ticker}...")
            cik = get_company_cik(ticker)
            if not cik:
                continue

            filings = fetch_8k_filing_links(cik)
            print(f"    -> Found {len(filings)} Form 8-K filings for {ticker}.")

            for filing in filings:
                if extracted_count >= TARGET_ENTRIES:
                    break

                text = fetch_filing_text(filing["link"])
                if not text:
                    continue

                # Skip parsing if keywords aren't present
                # (Comment this 'if' out if you want the LLM to scan every filing regardless)
                if not any(kw in text.lower() for kw in PRODUCT_KEYWORDS):
                    continue

                result = extract_product_with_llm(filing["company_name"], ticker, text)
                product = result.get("new_product", "None")

                # Only actual product found

                if product and product.lower() != "none":
                    writer.writerow([
                        result.get("company_name", filing["company_name"]),
                        result.get("stock_name", ticker),
                        result.get("filing_time", filing["filing_date"]),
                        product,
                        result.get("product_description", "None"),
                    ])
                    extracted_count += 1
                    print(f"        [✓] Saved entry {extracted_count}/{TARGET_ENTRIES} (Product: {product})")

    print(f"\n[✓] Pipeline complete! Successfully generated {extracted_count} rows in '{output_file}'.")


if __name__ == "__main__":
    main()