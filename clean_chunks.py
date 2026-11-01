import re
import json

def is_meaningful_english_chunk(text):
    # If more than 25% of the text is XBRL / XML / URLs / namespace tags, discard it
    if re.search(r'http://[^\s]+|fasb\.org|us-gaap|iso4217|xbrli', text):
        return False
    # Check if there are real English words
    words = re.findall(r'[a-zA-Z]{3,}', text)
    if len(words) < 15:
        return False
    # Check ratio of alphabetic characters
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / len(text) < 0.60:
        return False
    return True

with open('data/chunk_texts.json', 'r', encoding='utf-8') as f:
    raw_chunks = json.load(f)

filtered_chunks = []
for c in raw_chunks:
    if is_meaningful_english_chunk(c['text']):
        filtered_chunks.append(c)

print(f'Original chunks: {len(raw_chunks)}')
print(f'Filtered clean narrative chunks: {len(filtered_chunks)}')

with open('data/chunk_texts.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_chunks, f, indent=2)

print('Updated data/chunk_texts.json with clean text chunks.')
