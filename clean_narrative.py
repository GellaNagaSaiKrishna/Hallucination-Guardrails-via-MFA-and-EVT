import re
import json

def is_quality_narrative(text):
    # Skip XML/XBRL artefacts and code tags
    if any(k in text for k in ['http:', 'fasb.org', 'us-gaap', 'iso4217', 'xbrli', '&#', 'Member', 'Registrant', '0000320193']):
        return False
    # Must have complete sentences or coherent punctuation
    if not re.search(r'[.!?]', text):
        return False
    # Check word count
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 25:
        return False
    # Ensure high proportion of normal letters/spaces
    clean_ratio = sum(c.isalpha() or c in ' ,.!?;:\'"-()$%' for c in text) / len(text)
    if clean_ratio < 0.85:
        return False
    return True

with open('data/chunk_texts.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

narrative = [c for c in raw if is_quality_narrative(c['text'])]
print(f'Remaining high quality narrative text chunks: {len(narrative)}')

with open('data/chunk_texts.json', 'w', encoding='utf-8') as f:
    json.dump(narrative, f, indent=2)
