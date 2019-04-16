import pandas as pd
import sys

for a in sys.argv:
    raw = pd.read_csv(a, header=None, sep = "\t")
    shuffled = raw.sample(frac = 1)
    shuffled.to_csv(a.replace(".csv","_shuffled.csv"), header=False, sep = "\t", index=False)
