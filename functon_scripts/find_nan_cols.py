def find_nan_cols(data):
    nancols = dict()
    for column in data.columns:
        total_nan = sum(data[column].isnull())
        if total_nan > 0:
            nancols[column] = total_nan
    print(nancols)