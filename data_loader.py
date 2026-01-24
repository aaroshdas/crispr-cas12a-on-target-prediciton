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
    while len(list(df.columns)) > 2:
        column_list =list(df.columns)
        for c in column_list:
            #ADD FEATURE COLUMNS HERE TO KEEP
            if c != "Context Sequence" and c != "Indel frequency" and c != "gc content":
                df = df.drop(columns=[c], axis=1)

    df.rename(columns={"Context Sequence": "Input seq"}, inplace=True)
    df = df[df['Indel frequency'] >= -1]
    df['Indel frequency'] = df['Indel frequency'].clip(lower=0)
    return df
