"""Check LaTeX source consistency: environments and citations."""
import re

with open("paper_scs_framework.tex") as f:
    tex = f.read()

begins = re.findall(r'\\begin\{(\w+)\}', tex)
ends = re.findall(r'\\end\{(\w+)\}', tex)
from collections import Counter
b, e = Counter(begins), Counter(ends)
ok = True
for env in set(list(b.keys()) + list(e.keys())):
    if b[env] != e[env]:
        print(f'MISMATCH: {env} begin={b[env]} end={e[env]}')
        ok = False
if ok:
    print('All environments balanced')

refs = re.findall(r'\\bibitem', tex)
cites = set(re.findall(r'\\citet?\{(\w+)\}', tex))
bibkeys = set(re.findall(r'\\bibitem\[.*?\]\{(\w+)\}', tex))
print(f'Bibliography entries: {len(refs)}')
print(f'Unique citation keys used: {cites}')
missing = cites - bibkeys
unused = bibkeys - cites
if missing:
    print(f'MISSING bib entries: {missing}')
if unused:
    print(f'UNUSED bib entries: {unused}')
if not missing and not unused:
    print('All citations matched')

# Count tables
tables = re.findall(r'\\label\{tab:(\w+)\}', tex)
print(f'Tables: {len(tables)} -> {tables}')

# Count sections
secs = re.findall(r'\\(?:sub)*section\*?\{(.+?)\}', tex)
print(f'Sections/subsections: {len(secs)}')
for s in secs:
    print(f'  - {s}')
