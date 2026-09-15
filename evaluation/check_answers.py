import json

with open('evaluation/results_2026-03-27.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Show answer for Q1 and Q6 to see citation format
for r in data['results']:
    if r['id'] in [1, 6]:
        print(f"Q{r['id']} answer:")
        print(r['answer'])
        print()