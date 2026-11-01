import urllib.request
import json
import re
import os
import time

headers = {'User-Agent': 'AcademicResearchStudent cs328@univ.edu'}

filings = [
    {
        'ticker': 'AAPL', 'company': 'Apple Inc.', 'year': '2025',
        'url': 'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm'
    },
    {
        'ticker': 'MSFT', 'company': 'Microsoft Corp.', 'year': '2026',
        'url': 'https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm'
    },
    {
        'ticker': 'GOOGL', 'company': 'Alphabet Inc.', 'year': '2025',
        'url': 'https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm'
    },
    {
        'ticker': 'NVDA', 'company': 'NVIDIA Corp.', 'year': '2026',
        'url': 'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm'
    },
    {
        'ticker': 'AMZN', 'company': 'Amazon.com Inc.', 'year': '2026',
        'url': 'https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm'
    },
    {
        'ticker': 'META', 'company': 'Meta Platforms Inc.', 'year': '2026',
        'url': 'https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm'
    },
    {
        'ticker': 'TSLA', 'company': 'Tesla Inc.', 'year': '2026',
        'url': 'https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm'
    }
]

def clean_html(raw_html):
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', raw_html, flags=re.IGNORECASE)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'").replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    step = chunk_size - chunk_overlap
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) > 120:
            chunks.append(chunk)
        start += step
    return chunks

all_chunks = []
os.makedirs('data/raw', exist_ok=True)

for f in filings:
    ticker = f['ticker']
    raw_path = f'data/raw/{ticker}_10K.htm'
    
    if not os.path.exists(raw_path):
        print(f'Downloading {ticker} 10-K...')
        req = urllib.request.Request(f['url'], headers=headers)
        with urllib.request.urlopen(req, timeout=45) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
        with open(raw_path, 'w', encoding='utf-8') as out:
            out.write(content)
        time.sleep(1)
    else:
        print(f'Reading cached {ticker} 10-K...')
        with open(raw_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
            
    cleaned = clean_html(content)
    chunks = chunk_text(cleaned, chunk_size=500, chunk_overlap=50)
    print(f'{ticker}: {len(chunks)} chunks produced.')
    
    for idx, c in enumerate(chunks):
        all_chunks.append({
            'chunk_id': f'{ticker}_chunk_{idx:05d}',
            'ticker': ticker,
            'company': f['company'],
            'year': f['year'],
            'text': c,
            'char_count': len(c)
        })

output_file = 'data/chunk_texts.json'
with open(output_file, 'w', encoding='utf-8') as out:
    json.dump(all_chunks, out, indent=2)

print(f'\nFinished! Total chunks in corpus: {len(all_chunks)}. Saved to {output_file}')
