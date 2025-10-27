import pandas as pd

df = pd.read_csv("/storu/zkyang/AAA_MIL/data/loops_ground_truth/hiccups_loop_5k.bedpe", sep="\t")
df['chr1'] = 'chr' + df['chr1'].astype(str)
df['chr2'] = 'chr' + df['chr2'].astype(str)
df.to_csv("/storu/zkyang/AAA_MIL/data/loops_ground_truth/hiccups_loop_5k_change.bedpe", sep="\t", index=False)
