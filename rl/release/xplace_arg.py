"""(11-14) Print the --xplace value (eval.py / train_ppo.py) of a release directory: pickers/xboss-v3/xplace.json with its file names
resolved to absolute paths.   python tools/rl/release/xplace_arg.py <release dir>"""
import json, os, sys
d = os.path.join(os.path.abspath(sys.argv[1]), 'pickers', 'xboss-v3'); x = json.load(open(os.path.join(d, 'xplace.json')))
x['model'] = os.path.join(d, x['model'])
if (x.get('pool') or {}).get('gen'): x['pool']['gen'] = os.path.join(d, x['pool']['gen'])
print(json.dumps(x, separators=(',', ':')))
