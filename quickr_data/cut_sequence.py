import pandas as pd
file = './quickr_data/cleaner_seqs.csv'
new_file = './quickr_data/shortened_cleaner_seqs.csv'

raw_data_df = pd.read_csv(file)
data_df = pd.DataFrame()

shortened_seqs = []
for i in raw_data_df["Context Sequence"]:
    shortened_seqs.append(i[:34])
    # shortened_seqs.append(i[-34:])

data_df["Context Sequence"] = shortened_seqs

data_df.to_csv(new_file, index=False)