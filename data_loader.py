def filter_df(df):
    while len(list(df.columns)) > 2:
        column_list =list(df.columns)
        for c in column_list:
            if c != "Context Sequence" and c != "Indel frequency":
                df = df.drop(columns=[c], axis=1)
    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    return df


def filter_df_new_features(df):
    #ADD FEATURE COLUMNS HERE TO KEEP
    cols = ["Context Sequence", "Indel frequency", "gc content", "pam_prox_at_frac", "pos18_is_c"]
    
    while len(list(df.columns)) > len(cols):
        column_list =list(df.columns)
        for c in column_list:
            if c not in cols:
                df = df.drop(columns=[c], axis=1)

    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    return df
