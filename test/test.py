import pandas as pd

if __name__ == "__main__":

    df = pd.read_excel(io='test.xlsx')
    print(df)
    res1 = df.sort_values(by=['count'], ascending=False).groupby(['job']).head()
    print(res1)
    res2 = df.groupby(['job']).apply(lambda x: x.sort_values('count', ascending=False))
    print(res2)

