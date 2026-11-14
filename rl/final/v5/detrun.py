"""(A, 11-13) run a training script with torch deterministic algorithms (needs CUBLAS_WORKSPACE_CONFIG=:4096:8):
python detrun.py <script.py> [args...] — the script itself is unchanged (frozen tree)."""
import os, sys, runpy, torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
script = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
