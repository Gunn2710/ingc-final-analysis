import pandas as pd

raw_file = "nagpra_raw.txt"
out_file = "nagpra_data_full.csv"

rows = []

with open(raw_file, encoding="utf-8") as f:
    lines = f.read().strip().splitlines()

# First line is the header in your pasted data
data_lines = lines[1:]

for line in data_lines:
    line = line.strip()
    if not line:
        continue

    # Split by whitespace; last 3 tokens are numbers, the rest is the institution name
    parts = line.split()
    if len(parts) < 4:
        # just in case there are weird blank / broken lines
        continue

    percent_str = parts[-1]          # like "2%"
    made_str = parts[-2]             # like "175" or "1,522"
    not_str = parts[-3]              # like "7,936"
    institution_name = " ".join(parts[:-3])

    # Clean numeric fields: remove commas, strip % sign
    def clean_int(s: str) -> int:
        return int(s.replace(",", ""))

    def clean_pct(s: str) -> float:
        return float(s.strip("%"))

    try:
        remains_not = clean_int(not_str)
        remains_made = clean_int(made_str)
        pct_made = clean_pct(percent_str)
    except ValueError:
        # If anything fails to parse, you can print and debug
        print("Skipping line (parse error):", line)
        continue

    rows.append({
        "Institution": institution_name,
        "Remains_Not_Made_Available": remains_not,
        "Remains_Made_Available": remains_made,
        "Percent_Made_Available": pct_made
    })

df = pd.DataFrame(rows)
df.to_csv(out_file, index=False)

print(f"Wrote {len(df)} rows to {out_file}")
print(df.head())
