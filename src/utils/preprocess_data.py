import pandas as pd

def preprocess_data(X_numeric, X_category, numeric_transformer, category_transformer, label_encoders, index=None):
    X_numeric_preprocessed = numeric_transformer.transform(X_numeric)
    X_category_preprocessed = category_transformer.transform(X_category)

    X_category_preprocessed = pd.DataFrame(X_category_preprocessed, columns=X_category.columns, index=index)

    X_category_encoded = pd.DataFrame(index=index)
    for column, encoder in label_encoders.items():
        X_category_encoded[column] = encoder.transform(X_category_preprocessed[column])

    X_preprocessed = pd.concat(
        [pd.DataFrame(X_numeric_preprocessed, columns=X_numeric.columns, index=index), X_category_encoded],
        axis=1)

    return X_preprocessed
