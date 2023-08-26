from sklearn.model_selection import train_test_split

def train_val_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.15, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, test_set)

def split_features(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

def remove_label(df, label_name):
    return df.drop(label_name, axis=1)