##################################################################
"""
Author: wendy
1/3/2019

This script is for Data Exploration. some simple analysis of the data and features, plot some figures
and feature importance.
Input: "all-UG.csv"
"""
##################################################################
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime
import random
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter

pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth',10000)
pd.set_option('precision', 2)

def create_pivot_NUS_Summary_1(dataframe,index_list, column_list, values_list):
    table_nus_sum = pd.pivot_table(dataframe, index=index_list, columns=column_list,
                                   values=values_list, aggfunc='count',
                                   fill_value=0, margins=True)

    table_nus_sum = table_nus_sum.div(2.0)
    table = pd.concat(
        [d.append(d.sum().rename((k, 'Subtotal', ''))) for k, d in table_nus_sum.groupby('RESIDENCY')])
    table.index = pd.MultiIndex.from_tuples(table.index)
    table.drop(('MATRIC', 'All'), axis=1, inplace=True)
    table.drop(('All', 'Subtotal'), inplace=True)
    print(table)
    return table

def create_pivot_NUS_Summary_2(dataframe, index_list, column_list, values_list):
    table_nus_sum = pd.pivot_table(dataframe, index=index_list, columns=column_list,
                                   values=values_list, aggfunc='count',
                                   fill_value=0)
    table= table_nus_sum.div(2.0)
    # sort the order according to cohort year
    table_sorted = table.reset_index().groupby('RESIDENCY').apply(lambda x: x.sort_values('COHORT_YR', ascending=False)).set_index(index_list)

    print(table_sorted)
    return table_sorted

if __name__ == "__main__":

    # specification setting
    #FT_PT filter
    setFT_PT_filter = True
    FT_PT_value = 1 # 1 or 2 if it is True

    # loading the resale flat price data
    df = pd.read_csv('e:\\temp\\all-UG3.csv',low_memory=False)
    print(df.head(n=5))

    # check the general information
    df.info()
    #df.describe()
    #print(df.describe())

    # add new values
    df["year"], df["month"], df["day"] = df["TRAN_D"].str.split('-', 3).str
    df["year"] = df['year'].astype('int64')
    df["month"] = df['month'].astype('int64')
    df["day"] = df['day'].astype('int64')
    df["semester"] = np.where(df["month"]==9, 1, 2)
    df["year"] = np.where(df["semester"] == 2, df["year"]-1, df["year"])
    #df.loc[df["semester"] == 9, 'semester'] = 1

    # set date as index
    df['TRAN_D'] = pd.to_datetime(df['TRAN_D'])
    df = df.set_index('TRAN_D')

    #df=df['20150430':'20180401']
    df = df['20150430':]

    # select all the columns
    sel_columns = ["FACULTY",  "HON_NH", "COHORT_YR", "RESIDENCY", "EMPLID", "RNS", "MATRIC", "STATUS", "DEGREE", "year", "semester"]
    Rdf = df.loc[:, sel_columns]
    Rdf['FACULTY'] = Rdf['FACULTY'].astype('category')
    Rdf['HON_NH'] = Rdf['HON_NH'].astype('category')
    Rdf['COHORT_YR'] = Rdf['COHORT_YR'].astype('category')
    Rdf['RESIDENCY'] = Rdf['RESIDENCY'].astype('category')
    #Rdf['EMPLID'] = Rdf['EMPLID'].astype('category')
    Rdf['RNS'] = Rdf['RNS'].astype('category')
    Rdf['STATUS'] = Rdf['STATUS'].astype('category')
    Rdf['MATRIC'] = Rdf['MATRIC'].astype('category')
    Rdf['DEGREE'] = Rdf['DEGREE'].astype('category')

    Rdf = Rdf[Rdf['STATUS'].isin(['CURRENT', 'SUSPENDED', 'ON STUDENT EXCHANGE PROGRAMME'])]

#    print(Rdf.head(n=5))
#    print(Rdf.tail(n=5))
    #print(Rdf.describe())
    #print(Rdf.info())

    NUS_grouped = Rdf.groupby(['FACULTY'])

    index_list = ["RESIDENCY", "RNS"]
    column_list = ["year"]
    values_list = ["MATRIC"]
    NUS_group_missing = []
    with pd.ExcelWriter('output3.xlsx') as writer:
        for NUS_name, NUS_group in NUS_grouped:
            print(NUS_name)
            if len(NUS_group) > 0:
                #table_nus_sum_1 = create_pivot_NUS_Summary_1(NUS_group,index_list, column_list, values_list)

                index_list= ["RESIDENCY", "RNS", "COHORT_YR"]
                # NUS_group_sorted = NUS_group.groupby(['RESIDENCY']).apply(lambda x: x.sort_values('COHORT_YR', ascending=False))
                table_nus_sum_2 = create_pivot_NUS_Summary_2(NUS_group, index_list, column_list, values_list)
                #print("\n")

                table_nus_sum_2.to_excel(writer, sheet_name=NUS_name[:30])
                #new_df.to_excel(writer, sheet_name=NUS_name[:30])



