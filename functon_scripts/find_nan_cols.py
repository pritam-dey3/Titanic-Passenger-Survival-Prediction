def find_nan_cols(data):
    nancols = []
    for column in data.columns:
        if data[column].isnull().any():
            nancols.append(column)
    return nancols