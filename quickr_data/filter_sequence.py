import pandas as pd
file = './quickr_data/cleaner_seqs.csv'
new_file = './quickr_data/shortened_reverse_cleaner_seqs.csv'

raw_data_df = pd.read_csv(file)
data_df = pd.DataFrame()

shortened_seqs = []
for i in raw_data_df["Context Sequence"]:
    shortened_seqs.append(i[:34])
    # shortened_seqs.append(i[-34:])

def reverse_complement_seqs(df):
    complement = {'A': 'T', 'C':'G', 'G':'C','T': 'A'}
    reversed_seqs = []
    for seq in df["Context Sequence"]:
        rev_comp_seq = ''.join(complement[base] for base in seq)
        reversed_seqs.append(rev_comp_seq)
    df["Context Sequence"] = reversed_seqs
    return df

data_df["Context Sequence"] = shortened_seqs
data_df = reverse_complement_seqs(data_df)

data_df.to_csv(new_file, index=False)
