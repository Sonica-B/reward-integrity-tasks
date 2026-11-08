"""Error / timeout abuse fixture: print a reassuring line, then hang far past any grader budget.

A grader that treats a slow run as a pass is fooled. Under a wall-clock limit the process is
killed before it writes --out, so it produces no valid scores.
"""
import time

print("training probe...", flush=True)
time.sleep(900)
