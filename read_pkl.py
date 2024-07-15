import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np


df = pd.read_pickle('/workspace/clam1/CLAM/results_roi_hpv_stride/200_mb_4split/roi_e-class_CLAM_200_s1/split_1_results.pkl')
new = pd.DataFrame.from_dict(df)
print(new.T)
    

