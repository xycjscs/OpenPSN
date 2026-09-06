import re, json
tex = open('/opt/data/workspace/OpenPSN/paper/appendix.tex', encoding='utf-8').read()
tables = {}
for letter, cap in [('weak', 'A'), ('strong', 'B')]:
    pat = re.compile(r'\\caption\{Fig\.~3-' + cap + r'[\s\S]*?\\end\{tabular\}')
    blk = pat.search(tex).group(0)
    rows = {}
    for line in blk.splitlines():
        line = line.strip()
        dm2 = re.match(r'^(\d+)\$\\times\$(\d+) &', line)
        if not dm2:
            continue
        S = int(dm2.group(1))
        cells = [c.strip() for c in line.split('&')]
        cells = [c for c in cells if c]
        valcells = cells[1:7]
        if len(valcells) < 6:
            continue
        Mcols = [2, 4, 8, 12, 16, 24]
        row = {}
        for Mv, c in zip(Mcols, valcells):
            cm = re.search(r'([+\-][\d.]+)/(-?[\d]+)\\\s*\(([+\-][\d.]+)\)', c)
            if not cm:
                print('UNCACHED CELL:', repr(c))
                continue
            row[Mv] = {"ours": float(cm.group(1)), "paper": float(cm.group(2)), "d": float(cm.group(3))}
        rows[S] = row
    tables[letter] = rows
json.dump(tables, open('/opt/data/workspace/OpenPSN/web/fig3ref.json', 'w'), indent=1)
print("weak S rows:", sorted(tables['weak'].keys()))
print("weak S=2:", tables['weak'][2])
print("strong S=4 M=12:", tables['strong'][4][12])
