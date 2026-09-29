import json, glob
best = json.load(open('scoreboard_best.json'))
for lk in ['pressure', 'single']:
    wins = ties = losses = new = 0; L = []
    for f in sorted(glob.glob(f'results/{lk}__*.json')):
        r = json.load(open(f))
        if 'cr' not in r:
            continue
        pb = best.get(f"{lk}__{r['variable']}")
        if not pb:
            new += 1; continue
        if r['cr'] > pb[0] * 1.005: wins += 1
        elif r['cr'] >= pb[0] * 0.995: ties += 1; L.append(('~', r['variable'], round(r['cr'], 2), pb[0], r['config_short']))
        else: losses += 1; L.append(('X', r['variable'], round(r['cr'], 2), pb[0], r['config_short'], pb[2]))
    print(f"{lk}: done {wins+ties+losses+new}, wins {wins}, ties {ties}, losses {losses}, no-previous {new}")
    for l in L: print('   ', l)
