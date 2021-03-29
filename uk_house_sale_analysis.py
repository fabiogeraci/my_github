import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import sys
import glob
import requests

desired_width = 1200
pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

data_path = os.path.join(os.getcwd(), 'data')

urls = ['http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv',
        'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2019.csv',
        'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv']

# To save to a relative path.
for url in urls:
    if os.path.exists(os.path.join(data_path, url.split('/')[-1])):
        print('file exists')
    else:
        response = requests.get(url, stream=True)
        print(response.status_code)
        print("Downloading %s" % url.split('/')[-1])
        with open(os.path.join(data_path, url.split('/')[-1]), 'wb') as f:
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

all_files = sorted(glob.glob(data_path + "/pp*.csv"))
print(all_files)

all_dfs = []

for filename in all_files:
    pd_name = 'ppd_year_' + filename.split('.')[0][-4:]
    print(pd_name)
    pd_name = pd.read_csv(filename, index_col=None, header=0)
    all_dfs.append(pd_name)

for df in all_dfs:
    df.columns = ['Tuid', 'Price', 'DateOfTrasfer', 'Postcode', 'Prop_Type', 'Old/New', 'Duration', 'PAON', 'SAON', 'Street', 'Locality', 'Town/City',
                  'District', 'County', 'PPD', 'Record Status']

price_paid_data = pd.concat(all_dfs).reset_index(drop=True)
price_paid_data = price_paid_data.sort_values('DateOfTrasfer').reset_index(drop=True)

print(price_paid_data.head(10))
print(price_paid_data.info())
print(price_paid_data.isnull().sum())
print(price_paid_data.isnull().sum() * 100 / len(price_paid_data))
price_paid_data.drop(['Tuid','Old/New', 'Duration','SAON', 'Locality', 'Record Status'], axis=1, inplace=True)
price_paid_data = price_paid_data.dropna()
print(price_paid_data.isnull().sum() * 100 / len(price_paid_data))

def max_price_grp(df):
    """
    This function that will take price paid data and return another DataFrame containing the full details of the largest transaction
    occurring within each county present in the data.
    :param df:
    :return:
    """
    # grouping by County and finding the max price
    df = df.groupby('County')['Price'].max().reset_index()

    return df

max_price_per_county = max_price_grp(price_paid_data)
print(max_price_per_county.head())

def max_price_quarter(df):
    """
    This function that will take price paid data and return a DataFrame (indexed by quarter) giving the 5
    postcode districts (i.e. AB1 2CD => AB1) with the largest total transaction value for each quarter (and these values).
    :param df:
    :return:
    """

    df = df.copy()
    # to work only with dates (day)
    df.DateOfTrasfer = pd.to_datetime(df.DateOfTrasfer)
    # defining a column containing quarters Q1, Q2, Q3, Q4
    df['quarter'] = pd.PeriodIndex(df.DateOfTrasfer, freq='Q').astype(str).str[-2:]
    # split the postcode by space and returns the first part of the postcode
    df['Postcode'] = df.Postcode.str.split(' ', 0, expand=True)
    # grouping and calculating max value
    df = df.groupby(['quarter', 'District', 'Postcode'])['Price'].apply(np.max)

    # grouping quarter and district, returning the top 5
    df = df.groupby(level=[0, 1]).head(5).reset_index()
    # sorting the dataframe
    df.sort_values(['quarter', 'Price'], ascending=(True, False))

    df.set_index('quarter', inplace=True)

    return df

max_price_per_quarter = max_price_quarter(price_paid_data)
print(max_price_per_quarter)

def trans_value_conc(df):
    """
    This function that will take price paid data and return a DataFrame, indexed by year and with one column for each property type,
    giving the percentage of transactions (in descending order of size) that account for 80% of the total transaction value
    occurring for that property type for each year.
    :param df:
    :return:
    """

    df=df.copy()
    # to work only with dates (day)
    df.DateOfTrasfer = pd.to_datetime(df.DateOfTrasfer)
    # defining a column containing the year
    df['year'] = pd.PeriodIndex(df.DateOfTrasfer, freq='A')
    # format the values in the dataframe
    pd.options.display.float_format = '£{:,.2f}'.format
    # create a new column for each property type
    # calculates the 80% of the paid price
    df['pt_D'] = np.where(df['Prop_Type']=='D', df['Price']*0.8, 0)
    df['pt_S'] = np.where(df['Prop_Type']=='S', df['Price']*0.8, 0)
    df['pt_T'] = np.where(df['Prop_Type']=='T', df['Price']*0.8, 0)
    df['pt_F'] = np.where(df['Prop_Type']=='F', df['Price']*0.8, 0)
    df['pt_O'] = np.where(df['Prop_Type']=='O', df['Price']*0.8, 0)

    # df.drop('Prop_Type', axis=1)

    df = df.set_index('year')
    # grouping by index and calculating the total transaction value for each type
    df = df.groupby(df.index).agg({'pt_D':sum, 'pt_S':sum, 'pt_T':sum, 'pt_F':sum, 'pt_O':sum})

    return df

trans_sum = trans_value_conc(price_paid_data)
print(trans_sum)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def volume_median_price(df):
    """
    This function will take two subsets of price paid data and returns a DataFrame showing the percentage change in the number of transactions
    and their median price between the two datasets, broken down by each of the following price brackets:

    £0 < x <= £250,000
    £250,000 < x <= £500,000
    £500,000 < x <= £750,000
    £750,000 < x <= £1,000,000
    £1,000,000 < x <= £2,000,000
    £2,000,000 < x <= £5,000,000
    £5,000,000+
    The return value should be a DataFrame, indexed by price bracket expressed as a 2-tuple, and with columns for % change in transaction volume &
    % change in median price.

    Since only two subsets are required, the function is not adaptable if more subset are needed.
    :param df:
    :return:
    """
    # creating the two sets
    df_1 = df[1]
    df_2 = df[2]
    # price ranges
    bins = [0, 250000, 500000, 750000, 1000000, 2000000, 5000000, np.inf]
    # labels for index
    labels = [(0, 250000), (250000, 500000), (500000, 750000),
              (750000, 1000000), (1000000, 2000000),
              (2000000, 5000000), (5000000,)]
    # grouping and slicing by price ranges
    # calculating the total transactions and median price
    df_1 = df_1.groupby(pd.cut(df_1['Price'], bins=bins, labels=labels)).agg({'Postcode': 'size', 'Price': 'median'}) \
        .rename(columns={'Postcode': 'count_1', 'Price': 'Price_Median_1'})

    df_2 = df_2.groupby(pd.cut(df_2['Price'], bins=bins, labels=labels)).agg({'PPD': 'size', 'Price': 'median'}) \
        .rename(columns={'PPD': 'count_2', 'Price': 'Price_Median_2'})
    # concataneting into one dataframe
    df = pd.concat([df_1, df_2], axis=1)
    # calculating count and median price percentage between the subsets
    df['count_pct_change'] = (df.count_1 - df.count_2) / df.count_1
    df['Median_Pct_Change'] = (df.Price_Median_1 - df.Price_Median_2) / df.Price_Median_1

    cols = [0, 1, 2, 3]
    df.drop(df.columns[cols], axis=1, inplace=True)

    df['count_pct_change'] = df['count_pct_change'].map('{:.2%}'.format)
    df['Median_Pct_Change'] = df['Median_Pct_Change'].map('{:.2%}'.format)

    return df

vol_price_median = volume_median_price(all_dfs)
print(vol_price_median)

pd.reset_option('^display.', silent=True)
pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)


def holding_time(df):
    df = df.copy()
    # reduce df size
    df = df[['Price', 'DateOfTrasfer', 'Prop_Type', 'Postcode', 'PAON', 'Street']]
    # find duplicated address
    df = df[df.duplicated(subset=['Postcode', 'PAON', 'Street'], keep=False)]

    df.DateOfTrasfer = pd.to_datetime(df.DateOfTrasfer)
    # group by address calculate average price difference and average hold time
    df['avg_price'] = df.groupby(['Postcode', 'PAON', 'Street'])['Price'].transform(lambda x: x.diff().mean())
    df['avg_hold'] = df.groupby(['Postcode', 'PAON', 'Street'])['DateOfTrasfer'].transform(lambda x: x.diff().dt.days.mean())
    # drop duplicated line
    df.drop_duplicates(subset=['Postcode', 'PAON', 'Street'], keep='first', inplace=True)

    df.drop(['Price', 'DateOfTrasfer'], axis=1, inplace=True)

    df = df.dropna()

    df['avg_hold'] = df['avg_hold'].map('Days {:.1f}'.format)
    df['avg_price'] = df['avg_price'].map('£{:,.1F}'.format)

    return df


avg_hold_time_price = holding_time(price_paid_data)
print(avg_hold_time_price.head(5))

max = avg_hold_time_price['avg_hold'][avg_hold_time_price['avg_hold'].replace({r'Days ': r''}, regex=True).astype(float).idxmax()]
min = avg_hold_time_price['avg_hold'][avg_hold_time_price['avg_hold'].replace({r'Days ': r''}, regex=True).astype(float).idxmin()]
print('Max holding days'+str(max), 'Minimum holding days'+str(min))

adr_max = avg_hold_time_price.loc[avg_hold_time_price['avg_price'].replace(r'[£,]', r'', regex=True).astype(float).idxmax(), :]
adr_min = avg_hold_time_price.loc[avg_hold_time_price['avg_price'].replace(r'[£,]', r'', regex=True).astype(float).idxmin(), :]
print(adr_max, adr_min)
