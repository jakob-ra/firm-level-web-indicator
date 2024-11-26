import pandas as pd
from utils import remove_url_prefix
import awswrangler as wr
import numpy as np
from utils import get_currency_conversion_rates, get_nace_code_descriptions, get_nace_to_wk08_dict, \
    get_oxford_policy_tracker, country_codes_conversion
import requests
import json
from collections.abc import Iterable


## read and merge compustat global and us quarterly data
df_global = pd.read_csv('C:/Users/Jakob/Downloads/compustat-global-quarterly-1-5-23.csv')
df_global['source'] = 'global'
df_us = pd.read_csv('C:/Users/Jakob/Downloads/crsp-compustat-us-quarterly-1-05-23.csv') #'C:/Users/Jakob/Downloads/compustat-us-quarterly-25-4-23.csv')
df_us['source'] = 'NA'
df_us.rename(columns={'GVKEY': 'gvkey'}, inplace=True)

df = pd.concat([df_global, df_us], axis=0, ignore_index=True)
del df_us, df_global

## tidy
df.rename(columns={'weburl': 'url'}, inplace=True)
df = df.dropna(subset=['gvkey', 'url', 'fqtr', 'fyearq'])
# df = df[(df.fyearq >= 2018)] # &  (df.fyearq <= 2022)]
df['date'] = df.fyearq.astype(int).astype(str) + '-Q' + df.fqtr.astype(int).astype(str)
df.url = df.url.apply(remove_url_prefix)
df['datadate'] = pd.to_datetime(df.datadate)

## drop duplicate entries for same gvkey and date, keeping observations with most non-null financials, preferring Compustat North America
financial_cols = ['saleq', 'atq', 'gpq', 'revtq']
df['num_na'] = df[financial_cols].isna().sum(axis=1)
df['source_NA'] = df.source == 'NA'
df.sort_values(['loc', 'gvkey', 'date', 'source_NA', 'num_na', 'atq'], ascending=[True, True, True, False, False, False], inplace=True)
df.drop_duplicates(subset=['gvkey', 'date'], keep='first', inplace=True)
df.drop(columns=['num_na', 'source_NA'], inplace=True)

## export unique urls
df.url.drop_duplicates().to_csv('C:/Users/Jakob/Downloads/compustat-urls.csv', index=False, header=False)

## sic codes to NACE
naics_to_nace = pd.read_excel('https://www.census.gov/naics/concordances/2017_NAICS_to_ISIC_4.xlsx',
                              usecols=['2017\nNAICS\nUS  ', 'ISIC 4.0'])
naics_to_nace.columns = ['naics', 'nace']
naics_to_nace = naics_to_nace[naics_to_nace.naics != 0]
naics_to_nace['naics'] = naics_to_nace.naics.astype(float)
naics_to_nace_dict = naics_to_nace.set_index('naics').squeeze().to_dict()
df['nace'] = df.naics.map(naics_to_nace_dict)
na_mask = df.nace.isna()
unmapped_naics = df.loc[na_mask, 'naics'].unique()
for naics in unmapped_naics:
    nace = naics_to_nace[naics_to_nace.naics.astype(str).str.startswith(str(naics)[:-2])].nace.mode().values
    if len(nace) > 0:
        naics_to_nace_dict[naics] = nace[0]
df['nace'] = df.naics.map(naics_to_nace_dict)
df['nace_2_digit'] = df.nace.astype(str).str[:2]
df[df.nace_2_digit == 'na'][['naics', 'sic', 'nace']]

## NACE to 2-digit and section
nace, nace_sections_dict = get_nace_code_descriptions()
df['nace_section'] = df.nace_2_digit.map(nace_sections_dict)
# separate passenger air travel from other transport and storage
df.nace_section = df.nace_section.replace({'Transporting and storage': 'Transporting and storage (excl. passenger air transport)'})
df.loc[df.nace.apply(str).str.startswith('511'), 'nace_section'] = 'Passenger air transport'
df.groupby('nace_section', dropna=False).gvkey.nunique().sort_values(ascending=False)
df[na_mask].naics.value_counts().head(50)

## NACE to WK08
all_nace = df.nace.unique()
nace_to_wk08_dict = get_nace_to_wk08_dict(all_nace)
df['wk08'] = df.nace.map(nace_to_wk08_dict)
df.wk08.value_counts(dropna=False).head(50)
df[df.wk08.isna()].nace.astype(str).str[:2].value_counts().head(50)

## merge with Compustat daily security data
# do this to merge on quarter end dates
# all_quarter_closing_dates = df.datadate.value_counts().head(29).sort_index().reset_index(name='count')[
#     'index'].rename('datadate').to_frame()
# all_gvkeys = df.gvkey.unique()
# all_firm_dates = pd.merge(all_quarter_closing_dates, pd.DataFrame(all_gvkeys, columns=['gvkey']),
#                             how='cross')
# # read in chunks
# chunksize = 1e6
# cs_global_sec = pd.read_csv('C:/Users/Jakob/Downloads/compustat-global-daily-securities.csv',
#                             chunksize=chunksize, usecols=['gvkey', 'datadate', 'prccd', 'ajexdi', 'trfd'],
#                             parse_dates=['datadate'])
# chunks = []
# for i, chunk in enumerate(cs_global_sec):
#     print(f'Processing chunk {i}...')
#     chunk.drop_duplicates(subset=['datadate', 'gvkey'], keep='first', inplace=True)
#     chunk.sort_values(['datadate', 'gvkey'], inplace=True)
#     chunk = pd.merge_asof(all_firm_dates, chunk, on='datadate',
#                           direction='backward', tolerance=pd.Timedelta('7 days'), by='gvkey')
#     chunk.dropna(subset=['prccd'], inplace=True)
#     chunks.append(chunk)
#
# cs_na_sec = pd.read_csv('C:/Users/Jakob/Downloads/compustat-na-daily-securities.csv',
#                         chunksize=chunksize, usecols=['gvkey', 'datadate', 'prccd', 'ajexdi', 'trfd'],
#                         parse_dates=['datadate'])
# for i, chunk in enumerate(cs_na_sec):
#     print(f'Processing chunk {i}...')
#     chunk.drop_duplicates(subset=['datadate', 'gvkey'], keep='first', inplace=True)
#     chunk.sort_values(['datadate', 'gvkey'], inplace=True)
#     chunk = pd.merge_asof(all_firm_dates, chunk, on='datadate',
#                           direction='backward', tolerance=pd.Timedelta('7 days'), by='gvkey')
#     chunk.dropna(subset=['prccd'], inplace=True)
#     chunks.append(chunk)
#
# cs_sec = pd.concat(chunks).drop_duplicates(subset=['datadate', 'gvkey'])
# cs_sec.to_parquet('C:/Users/Jakob/Downloads/compustat-daily-securities.parquet')
cs_sec = pd.read_parquet('C:/Users/Jakob/Downloads/compustat-daily-securities.parquet')

df = df.merge(cs_sec, on=['gvkey', 'datadate'], how='left')


# the formula for returns is ((prccd/ ajexdi )* trfd )[t] /( prccd/ ajexdi )* trfd ))[t-1]-1)*100)
df['stock_closing'] = (df.prccd / df.ajexdi) * df.trfd
df['returnq'] = df.groupby('gvkey').stock_closing.pct_change(fill_method=None) * 100
df.loc[(df.returnq == np.inf) | (df.returnq == -np.inf), 'returnq'] = np.nan
df.loc[df.returnq > df.returnq.quantile(0.999), 'returnq'] = np.nan # set returns above 99.9 percentile to nan
# df.returnq = df.returnq.clip(upper=df.returnq.quantile(0.999)) # clip returns at 99.9 percentile
df['log_stock_closing'] = np.log(df.stock_closing)
df['log_diff_stock_closing_pct'] = df.groupby('gvkey').log_stock_closing.diff() * 100
df.loc[(df.log_diff_stock_closing_pct == np.inf) | (df.log_diff_stock_closing_pct == -np.inf), 'log_diff_stock_closing_pct'] = np.nan
# df['log_diff_stock_closing_pct_no_outliers'] = df['log_diff_stock_closing_pct']
# df.loc[df.log_diff_stock_closing_pct_no_outliers > df.log_diff_stock_closing_pct_no_outliers.quantile(0.999), 'log_diff_stock_closing_pct_no_outliers'] = np.nan
# df.loc[df.log_diff_stock_closing_pct_no_outliers < df.log_diff_stock_closing_pct_no_outliers.quantile(0.001), 'log_diff_stock_closing_pct_no_outliers'] = np.nan

df.returnq.describe()
## explore
quarter_counts = df.date.value_counts().sort_index()
df.curcdq.value_counts().head(50)
df.groupby('source').gvkey.nunique()

## convert non USD financials to USD
# get daily exchange rates from yahoo finance
currencies = df.curcdq.value_counts().index.tolist()
currencies.remove('USD')
quarter_report_dates = df[['date', 'datadate']].value_counts().sort_index().reset_index()
start_date = quarter_report_dates.datadate.min()
end_date = quarter_report_dates.datadate.max()
conversion_rates = get_currency_conversion_rates(currencies, start_date, end_date)
conversion_rates.rename(columns={'close': 'conversion_rate', 'symbol': 'curcdq', 'date': 'datadate'}, inplace=True)
conversion_rates['datadate'] = pd.to_datetime(conversion_rates.datadate)

df = df.merge(conversion_rates, how='left', on=['curcdq', 'datadate'], validate='m:1')

# df['saleq'] = df.saleq.fillna(df.revtq)
df.loc[df.saleq < 0, 'saleq'] = np.nan # df['saleq'] = df.saleq.clip(lower=0) decide whether to clip or set na

for col in financial_cols:
    df[col] = df[col] * 1e6 # Financials are scaled in millions
    df[f'{col}_usd'] = np.nan
    usd_mask = df.curcdq == 'USD'
    df.loc[usd_mask, f'{col}_usd'] = df.loc[usd_mask, col]
    df.loc[~usd_mask, f'{col}_usd'] = df.loc[~usd_mask, col] / df.loc[~usd_mask, 'conversion_rate']

## calculate quarterly growth rates for financials (fill_method=None is necessary to avoid filling with 0 when var is NaN)
for col in financial_cols + [col + '_usd' for col in financial_cols]:
    df[f'{col}_gr'] = df.groupby('gvkey')[col].pct_change(fill_method=None) * 100 # calculate percentage change
    df[f'{col}_gr'] = df[f'{col}_gr'].replace([np.inf, -np.inf], np.nan) # replace inf with NaN
    # df[f'{col}_gr'] = df[f'{col}_gr'].clip(upper=df[f'{col}_gr'].quantile(0.99)) # clip growth rates at 99th percentile
    # set returns above 99.9 percentile to nan
    df.loc[df[f'{col}_gr'] > df[f'{col}_gr'].quantile(0.999), f'{col}_gr'] = np.nan

df.saleq_usd_gr.describe()

# there are many URLs that are not unique to one firm - e.g. ishares.com, fidelity.com, etc.
df.groupby('url').gvkey.nunique().sort_values(ascending=False).head(50)

# we need to filter out firms with these URLs
urls_unique_to_one_firm = df.drop_duplicates(subset=['gvkey']).url.drop_duplicates(keep=False)
df = df[df.url.isin(urls_unique_to_one_firm)]

df.url.nunique()

## export list of unique URL
# wr.s3.to_csv(df.url.drop_duplicates(), 's3://cc-download-compustat/url-list/compustat_unique_urls.csv', index=False, header=False)

# get retuns data from yahoo finance
#
# def get_symbol_for_isin(isin):
#     url = 'https://query1.finance.yahoo.com/v1/finance/search'
#
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.109 Safari/537.36',
#     }
#
#     params = dict(
#         q=isin,
#         quotesCount=1,
#         newsCount=0,
#         listsCount=0,
#         quotesQueryId='tss_match_phrase_query'
#     )
#
#     try:
#         resp = requests.get(url=url, headers=headers, params=params)
#         data = resp.json()
#         if 'quotes' in data and len(data['quotes']) > 0:
#             return data['quotes'][0]['symbol']
#         else:
#             return None
#     except Exception as e:
#         print(e)
#         return None
#
# sample = df[df['isin'].notna()][['isin', 'datadate']].sample().iloc[0]
# isin = sample['isin']
# print(isin)
# print(get_symbol_for_isin(isin))
#
# (df.groupby('gvkey')['tic'].count() > 0).sum()
#
# from tqdm import tqdm
# tqdm.pandas()
#
# df['isin'].value_counts(dropna=False).head(50)
# isins = list(df.loc[df['isin'].str.len() > 5, 'isin'].unique())
# isin_to_tic = {}
# for isin in tqdm([isin for isin in isins if not isin in isin_to_tic]):
#     isin_to_tic[isin] = get_symbol_for_isin(isin)
#
# df.loc[df['tic'].isna(), 'tic'] = df.loc[df['tic'].isna(), 'isin'].map(isin_to_tic)
#
# from yahooquery import Ticker
#
# def get_adjusted_close_price(isin, date, max_bfill=30):
#     # Format the date to match the Yahoo Finance API requirements
#     date_str = date.strftime('%Y-%m-%d')
#     date_str_plus_one = (date + pd.DateOffset(days=max_bfill)).strftime('%Y-%m-%d')
#
#     # Retrieve the ticker data using the ISIN
#     ticker = Ticker(isin)
#
#     # Retrieve the adjusted historical prices for the specified date
#     hist_data = ticker.history(start=date_str, end=date_str_plus_one, adj_ohlc=True, interval='1d')
#
#     try:
#         return hist_data['close'][0] # returns the date closest to the specified date
#     except:
#         return np.nan
#
# sample = df[df.tic.notna()][['tic', 'datadate']].sample().iloc[0]
# print(sample.tic)
# print(sample.datadate)
# get_adjusted_close_price(sample.tic, sample.datadate)
#
# get_adjusted_close_price('AAPL', pd.to_datetime('2017-01-09'))

## country to world region
path = 'https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv?raw=true'
countries = pd.read_csv(path, sep=',', usecols=['name', 'region', 'sub-region', 'alpha-2', 'alpha-3'])
countries = countries.rename(columns={'name': 'country', 'alpha-2': 'country_iso2', 'alpha-3': 'country_iso3'})
df = df.merge(countries, left_on='loc', right_on='country_iso3', how='left')

## read cc results
crawl_times = pd.read_csv('/Users/jakob/Downloads/common-crawls.csv', sep=';')
crawl_quarter_map = {'CC-MAIN-2020-05': '2020-Q1', 'CC-MAIN-2020-10': '2020-Q1',
                     'CC-MAIN-2020-16': '2020-Q2', 'CC-MAIN-2020-24': '2020-Q2',
                     'CC-MAIN-2020-29': '2020-Q3', 'CC-MAIN-2020-34': '2020-Q3',
                     'CC-MAIN-2020-40': '2020-Q3', 'CC-MAIN-2020-45': '2020-Q4',
                     'CC-MAIN-2020-50': '2020-Q4', 'CC-MAIN-2021-04': '2021-Q1',
                     'CC-MAIN-2021-10': '2021-Q1', 'CC-MAIN-2021-17': '2021-Q2',
                     'CC-MAIN-2021-21': '2021-Q2', 'CC-MAIN-2021-25': '2021-Q2',
                     'CC-MAIN-2021-31': '2021-Q3', 'CC-MAIN-2021-39': '2021-Q3',
                     'CC-MAIN-2021-43': '2021-Q4', 'CC-MAIN-2021-49': '2021-Q4',
                     'CC-MAIN-2022-05': '2022-Q1', 'CC-MAIN-2022-21': '2022-Q2',
                     'CC-MAIN-2022-27': '2022-Q2', 'CC-MAIN-2022-33': '2022-Q2',
                     'CC-MAIN-2022-33': '2022-Q3', 'CC-MAIN-2022-40': '2022-Q3',
                     'CC-MAIN-2022-49': '2022-Q4', 'CC-MAIN-2023-06': '2023-Q1',
                     'CC-MAIN-2023-14': '2023-Q2'}
crawl_quarter_map = pd.Series(crawl_quarter_map).to_frame('date').reset_index(names='crawl')
crawl_times = crawl_times.rename(columns={'Name': 'crawl'}).merge(crawl_quarter_map, on='crawl', how='left')

# from problem_classification_sentiment import process_and_save
# project_name = 'cc-download-compustat'
# topic_keywords_path = 'https://github.com/jakob-ra/cc-download-translate/raw/main/topic_keywords.json'
# cc, _ = process_and_save(project_name, topic_keywords_path)

# cc = wr.s3.read_parquet(
#     f's3://{project_name}/classified/' + 'paragraph_level/' + project_name + '_classified_paragraphs.parquet')
# cc.to_parquet('C:/Users/Jakob/Downloads/cc_classified.parquet')
cc = pd.read_parquet('C:/Users/Jakob/Downloads/cc_classified.parquet')

# fix dates (quarterly data)
domains_per_crawl = cc.groupby('crawl').url_host_registered_domain.nunique()

# merge with dates
cc = cc.merge(crawl_quarter_map, on='crawl', how='left')
cc.groupby('date').url_host_registered_domain.nunique()

# group by domain and date
labels = ['production', 'demand', 'supply', 'travel', 'finance']
neg_sent_labels = [f'{label}_neg_sent' for label in labels]
agg_dict = {label: 'sum' for label in labels + neg_sent_labels} | {'sentiment': 'mean'}
cc_grouped = cc.groupby(['url_host_registered_domain', 'date'])[labels + neg_sent_labels + ['sentiment']].agg(agg_dict)
cc_grouped['covid_mention'] = (cc.paragraph.apply(str).str.len() > 0).groupby([
        cc.url_host_registered_domain, cc.date]).sum()

# export covid mention separately
# covid_mention = (cc.paragraph.apply(str).str.len() > 0).groupby([cc.url_host_registered_domain, cc.date]).sum()
# covid_mention = covid_mention > 0
# covid_mention = covid_mention.to_frame(name='covid_mention').reset_index()
# covid_mention.rename(columns={'url_host_registered_domain': 'url'}, inplace=True)
# covid_mention['date'] = pd.to_datetime(covid_mention.date).dt.to_period('Q').astype(str)
# covid_mention.set_index(['url', 'date'], inplace=True)
# covid_mention.to_stata('C:/Users/Jakob/Downloads/covid_mention.dta')

# # llm results
# cc_llm = wr.s3.read_parquet('s3://cc-download-compustat-new/res_llm_consolidated/res_llm_new_prompt_without_texts_llama_3_1.parquet')
# cc_llm.to_pickle('C:/Users/Jakob/Downloads/cc_llm_llama_3_1.pkl')
# cc_llm = pd.read_pickle('C:/Users/Jakob/Downloads/cc_llm_llama_3_1.pkl')

def combine_tags(tags, tag_mapping):
    if isinstance(tags, Iterable):
        new_tags = []
        for t in tags:
            if t.strip()=='':
                continue
            combined = False
            for new_tag, keywords in tag_mapping.items():
                if any(keyword in t for keyword in keywords):
                    new_tags.append(new_tag)
                    combined = True
            if not combined:
                new_tags.append('other')
        return list(set([t.strip() for t in new_tags]))

    return tags

experiment_labels = {'res_llm_new_prompt_without_texts_llama_3_1_manual_logits4.parquet': 'llama',
                     'res_llm_new_prompt_without_texts_gpt.parquet'                     : 'gpt'}

# res = wr.s3.read_parquet(f's3://cc-download-compustat-new/res_llm_consolidated/res_llm_new_prompt_without_texts_llama_3_1_manual_logits4.parquet')
# res['tags_combined'] = res.tags.apply(lambda x: combine_tags(x, tag_mapping))
# res['tags_combined'].explode().value_counts().head(50)

with open('tag_mapping_new.json', 'r') as f:
    tag_mapping = json.load(f)

for res_path in experiment_labels.keys():
    res = wr.s3.read_parquet(f's3://cc-download-compustat-new/res_llm_consolidated/{res_path}')
    res = res.merge(crawl_quarter_map, on='crawl', how='left')

    res['tags_combined'] = res.tags.apply(lambda x: combine_tags(x, tag_mapping))

    res['production_affected'] = res.affectedness_category.apply(lambda x: 'production' in x)
    res['demand_affected'] = res.affectedness_category.apply(lambda x: 'demand' in x)
    res['supply_affected'] = res.affectedness_category.apply(lambda x: 'supply' in x)

    tags = ['hygiene measures', 'remote work', 'supply chain issues', 'closure', 'other']

    for tag in tags:
        res[tag] = res.tags_combined.apply(lambda x: tag in x)

    affected_vars = ['affected']
    agg_dict = {k: 'max' for k in affected_vars}
    agg_dict.update({tag: 'max' for tag in tags})
    affectedness_categories = ['production_affected', 'demand_affected', 'supply_affected']
    agg_dict.update({cat: 'max' for cat in affectedness_categories})

    # paragraph expiry for llama results
    # if res_path == 'res_llm_new_prompt_without_texts_llama_3_1_manual_logits3.parquet':
    #     affected_expiry_vars = []
    #     for month_limit in [0, 3, 6, 12]:
    #         affected_expiry_var = f'affected_w_expiry_{month_limit}'
    #         res[affected_expiry_var] = res.affected * (res.months_unchanged < month_limit)
    #         affected_expiry_vars.append(affected_expiry_var)
    #     agg_dict.update({affected_expiry_var: 'max' for affected_expiry_var in affected_expiry_vars})

    # use paragraph expiry of 12 months for all indicators
    month_limit = 12
    if res_path == 'res_llm_new_prompt_without_texts_llama_3_1_manual_logits3.parquet':
        for col in affected_vars + affectedness_categories + tags:
            res[col] = res[col] * (res.months_unchanged < month_limit)

    res_grouped = res.groupby(['url_host_registered_domain', 'date']).agg(agg_dict)

    res_grouped.columns = [f'{col}_{experiment_labels[res_path]}' for col in res_grouped.columns]
    cc_grouped = cc_grouped.join(res_grouped, how='outer')

    del res, res_grouped

cc_grouped.reset_index(inplace=True)
cc_grouped.rename(columns={'url_host_registered_domain': 'url'}, inplace=True)

# expiry:
# for month_limit in [3,6,12]:
#     df_res_llm[f'affected_w_expiry_{month_limit}'] = df_res_llm.affected*(df_res_llm.months_unchanged<month_limit)
# import matplotlib.pyplot as plt
# df_res_llm[df_res_llm['affected']>0].groupby('fetch_time').url_host_registered_domain.nunique().plot(label=0)
# for month_limit in [3,6,12]:
#     df_res_llm[df_res_llm[f'affected_w_expiry_{month_limit}']>0].groupby('fetch_time').url_host_registered_domain.nunique().plot(label=month_limit)
# plt.legend()

# merge with compustat data
df = df.merge(cc_grouped, on=['url', 'date'], how='left', validate='1:1', indicator=True)
df['_merge'] = df['_merge'].map({'left_only': 0, 'both': 1})
merged_by_url = df.groupby('url')._merge.max()
merged_counts = merged_by_url.value_counts()
print(f'Found {merged_counts.loc[1.0]} of {merged_counts.sum()} Compustat URLs in CC data ({merged_counts.loc[1.0] / merged_counts.sum() * 100:.2f}%)')
df = df[df.url.isin(merged_by_url[merged_by_url == 1].index)] # drop URLs that are not in CC data
df.rename(columns={'_merge': 'cc_data_available'}, inplace=True) # this is only 1 for the pandemic years

# convert labels to dummy
for var in labels + neg_sent_labels + ['covid_mention']:
    df[var] = df[var].apply(lambda x: x > 0).fillna(0).astype(int)

df['any_neg_sent'] = df[neg_sent_labels].sum(axis=1) > 0
df['any_neg_sent'] = df['any_neg_sent'].astype(int)

df['date'] = pd.to_datetime(df.date).dt.to_period('Q')

df['log_saleq_usd'] = np.log(df.saleq_usd + 1)
df['log_atq_usd'] = np.log(df.atq_usd + 1)
df['l1_log_atq_usd'] = df.groupby('gvkey').log_atq_usd.shift(1)
df['log_diff_saleq_pct'] = df.groupby('gvkey').log_saleq_usd.diff() * 100

## data exploration
df[df.any_neg_sent>0].gvkey.nunique()
df.sort_values('gpq', ascending=False)[['conm', 'date', 'gpq']].head(20)
df[df.source == 'us'].groupby(['conm', 'date']).saleq.sum().sort_values(ascending=False).head(20)
df.groupby(['conm', 'date']).saleq.sum().sort_values(ascending=False).head(20)

## add oxford policy tracker data
oxford_policy_tracker = get_oxford_policy_tracker(aggregation_period='Q', extended_cols=True)
oxford_policy_tracker.rename(columns={'CountryCode': 'loc', 'Date': 'date'}, inplace=True)
oxford_policy_tracker.drop(columns=['StringencyIndex_Average_ForDisplay'], inplace=True)
oxford_policy_tracker_cols = [col for col in oxford_policy_tracker.columns if col not in ['loc', 'date']]
# center and normalize all index columns
index_cols = ['StringencyIndex_Average', 'GovernmentResponseIndex_Average',
                  'ContainmentHealthIndex_Average', 'EconomicSupportIndex']
# oxford_policy_tracker[index_cols] = (oxford_policy_tracker[index_cols] - oxford_policy_tracker[
#     index_cols].mean()) / oxford_policy_tracker[index_cols].std() # normalize index columns
## convert fiscal measures to % of GDP
path = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv'
# if down use '/Users/jakob/Downloads/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1584277.zip'
from zipfile import ZipFile
from io import BytesIO
with ZipFile(BytesIO(requests.get(path).content)) as zf:
    f = [f for f in zf.namelist() if f.startswith('API_NY')][0]
    with zf.open(f) as f:
        gdp_df = pd.read_csv(f, header=2)
gdp_df.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code'], inplace=True)
gdp_df = pd.wide_to_long(gdp_df, "", i="Country Code", j="year")
gdp_df.columns = ['', 'gdp']
gdp_df = gdp_df[['gdp']]
# ffill missing values
gdp_df['gdp'] = gdp_df.groupby('Country Code').gdp.ffill()
gdp_df.reset_index(inplace=True)
gdp_df = gdp_df[gdp_df.year == 2020].drop(columns=['year'])
oxford_policy_tracker = oxford_policy_tracker.merge(gdp_df, left_on='loc', right_on='Country Code', how='left').drop(columns=['Country Code'])
oxford_policy_tracker['fiscal_measures_pct_gdp'] = oxford_policy_tracker['E3_Fiscal measures'] / oxford_policy_tracker['gdp'] * 100
oxford_policy_tracker.sort_values('gdp', ascending=False).head(20)
oxford_policy_tracker_cols += ['fiscal_measures_pct_gdp']
# round all ordinal scale variables to integers
scale_vars = ['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport',
              'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief']
oxford_policy_tracker[scale_vars] = oxford_policy_tracker[scale_vars].round(0)
df = df.merge(oxford_policy_tracker, on=['loc', 'date'], how='left', validate='m:1')
oxford_policy_tracker_countries = set(oxford_policy_tracker['loc'].unique())
# fill na with 0 for all but the non-merged countries
df.loc[df['loc'].isin(oxford_policy_tracker_countries), oxford_policy_tracker_cols] = df.loc[
    df['loc'].isin(oxford_policy_tracker_countries), oxford_policy_tracker_cols].fillna(0)
# center and normalize all index columns AFTER filling na
index_cols = ['StringencyIndex_Average', 'GovernmentResponseIndex_Average',
                  'ContainmentHealthIndex_Average', 'EconomicSupportIndex']
df[index_cols] = (df[index_cols] - df[
    index_cols].mean()) / df[index_cols].std() # normalize index columns


# geocode city names
city_df = 'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/exports/csv?lang=en&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B'
city_cols = ['Name', 'ASCII Name', 'Alternate Names', 'Country Code', 'Admin1 Code', 'Admin2 Code',
             'Admin3 Code', 'Admin4 Code', 'Population', 'Elevation', 'Coordinates']
city_df = pd.read_csv(city_df, sep=';', usecols=city_cols)
city_df['num_na'] = city_df.isna().sum(axis=1)
city_df.sort_values('num_na', ascending=True, inplace=True)
city_df.drop_duplicates(subset=['Name', 'ASCII Name', 'Alternate Names'], keep='first', inplace=True)
city_df.drop(columns='num_na', inplace=True)

city_names = set(df.city.str.lower().str.strip().unique())

def get_merge_name(row, city_names):
    try:
        name = row['Name'].lower().strip()
        if name in city_names:
            return name
    except:
        return np.nan
    try:
        ascii_name = row['ASCII Name'].lower().strip()
        if ascii_name in city_names:
            return ascii_name
    except:
        return np.nan

    try:
        alternate_names = row['Alternate Names'].lower().strip().split(',')
        for alternate_name in alternate_names:
            if alternate_name in city_names:
                return alternate_name
    except:
        return np.nan

    return np.nan

city_df['merge_name'] = city_df.apply(lambda row: get_merge_name(row, city_names), axis=1)
city_df.drop(columns=['Name', 'ASCII Name', 'Alternate Names'], inplace=True)
city_df = city_df.dropna(subset=['merge_name']).drop_duplicates(subset=['merge_name'], keep='first')

df = df.merge(city_df, left_on=[df.city.str.lower().str.strip(), 'country_iso2'], right_on=['merge_name', 'Country Code'], how='left')

df['Admin1 Code'].value_counts(dropna=False).head(50)

# using geonames
# city_country = df[df['Admin1 Code'].isna()][['city', 'country_iso2']].drop_duplicates()
# import geocoder
# from tqdm import tqdm
# tqdm.pandas()
# city_country.sample(10).progress_apply(lambda x: geocoder.geonames(x, key='jrauch').json)
# geocoder.geonames('Dayton, USA', key='jrauch').json
# geocoder.geonames(city_country.iloc[1000], key='jrauch').json
#
# city_country['res'] = np.nan
# for i in tqdm(range(0, 1000)):
#     city = city_country.iloc[i]['city']
#     country = city_country.iloc[i]['country_iso2']
#     city_country.iloc[i]['res'] = geocoder.geonames(city, country=country, key='jrauch').json

df['Country-Region'] = np.nan
df.loc[df[['loc', 'Admin1 Code']].notna().any(axis=1), 'Country-Region'] = df.loc[df[['loc', 'Admin1 Code']].notna().any(axis=1), 'loc'].astype(str) + ' - ' + df.loc[df[['loc', 'Admin1 Code']].notna().any(axis=1), 'Admin1 Code'].astype(str)
df.loc[df['Country-Region'].str.contains(' - nan'), 'Country-Region'] = np.nan

df['Country-Region'].value_counts(dropna=False).head(50)

df.groupby('date').saleq_gr.mean()
df['loc'].value_counts().head(50)

df.nace_section.value_counts(dropna=False)

# add US state level policies
compustat_states = pd.read_csv('/Users/jakob/Downloads/compustat-na-states.csv')
compustat_states = compustat_states[['gvkey', 'state']]
compustat_states = compustat_states.drop_duplicates().dropna()
df = df.merge(compustat_states, on='gvkey', how='left')

# get regional oxford policy tracker data for US
from utils import get_oxford_policy_tracker_regional
oxford_policy_tracker_us = get_oxford_policy_tracker_regional('United States', aggregation_period='Q', extended_cols=True)

# get table of US state names to two-letter abbreviations
us_state_abbrev = pd.read_csv('https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv')
oxford_policy_tracker_us['state'] = oxford_policy_tracker_us['RegionName'].map(us_state_abbrev.set_index('State')['Abbreviation'])

oxford_policy_tracker_us.drop(columns=['StringencyIndex_Average_ForDisplay'], inplace=True)

# round all ordinal scale variables to integers
scale_vars = ['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport',
              'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief']
oxford_policy_tracker_us[scale_vars] = oxford_policy_tracker_us[scale_vars].round(0)

# merge with US data
oxford_policy_tracker_us.columns = [col + '_state' for col in oxford_policy_tracker_us.columns]
oxford_policy_tracker_us.rename(columns={'Date_state': 'date', 'state_state': 'state'}, inplace=True)
df = df.merge(oxford_policy_tracker_us, on=['state', 'date'], how='left', validate='m:1')

oxford_policy_tracker_states = set(oxford_policy_tracker_us['state'].dropna().unique())

oxford_policy_tracker_us_cols = [col for col in oxford_policy_tracker_us.columns if col not in ['state', 'date']]
# fill na with 0 for all but the non-merged states
df.loc[df['state'].isin(oxford_policy_tracker_states), oxford_policy_tracker_us_cols] = df.loc[
    df['state'].isin(oxford_policy_tracker_states), oxford_policy_tracker_us_cols].fillna(0)

# center and normalize all index columns AFTER filling na
index_cols = ['StringencyIndex_Average', 'GovernmentResponseIndex_Average',
                  'ContainmentHealthIndex_Average', 'EconomicSupportIndex']
index_cols_state = [col + '_state' for col in index_cols]
df[index_cols_state] = (df[index_cols_state] - df[
    index_cols_state].mean()) / df[index_cols_state].std() # normalize index columns

# add remote work data
remote_work = pd.read_excel('/Users/jakob/Downloads/remote_work_in_job_ads_signup_data.xlsx',
                            sheet_name='us_ind_by_month')
remote_work = remote_work[['Year', 'NAICS 2022 3-Digit Industry Group', 'Percent']]
remote_work.columns = ['year', 'naics_3_digit', 'remote_work_pct']
remote_work = remote_work[remote_work.year<2020]
remote_work = remote_work.groupby(['year', 'naics_3_digit']).remote_work_pct.median().reset_index()
remote_work.naics_3_digit = remote_work.naics_3_digit.astype(str)
df['naics_3_digit'] = df.naics.apply(str).str[:3]
df = df.merge(remote_work.drop(columns=['year']), on='naics_3_digit', how='left')
df[df.remote_work_pct.notna()].gvkey.nunique()

# add software costs
software_costs = pd.read_csv('/Users/jakob/Downloads/compustat-na-software-costs.csv')
software_costs.dropna(subset=['capsft'], inplace=True)
software_costs = software_costs[software_costs.fyear < 2020]
software_costs = software_costs.sort_values('fyear').drop_duplicates(subset=['gvkey'], keep='last') # take most recent observation for each gvkey
df = df.merge(software_costs[['gvkey', 'capsft']], on='gvkey', how='left')

# add firm-level wfh index
wfh_path = '/Users/jakob/Downloads/wfh_index_2010Q1_2020Q1.dta' # 2121 firms in sample
wfh_index = pd.read_stata(wfh_path)
wfh_index = wfh_index[wfh_index.year.between(2010,2019)].groupby('gvkey').wfh_index_qtr.mean().reset_index()
wfh_index['wfh_index_top_quartile'] = wfh_index.wfh_index_qtr > wfh_index.wfh_index_qtr.quantile(0.75).astype(float)
df = df.merge(wfh_index.rename(columns={'wfh_index_qtr': 'wfh_index'}), on='gvkey', how='left')
df.wfh_index_top_quartile = df.wfh_index_top_quartile.astype(float)

# dummy for 2020 q1-q3
df['covid_period_dummy'] = df.date.isin(['2020Q1', '2020Q2', '2020Q3'])

# add returns matched to exact fetch times
# fetch times of crawls
# fetch_times = wr.athena.read_sql_query(
#         """SELECT DISTINCT fetch_time, crawl FROM res WHERE fetch_time IS NOT NULL""",
#         database='ccindex')
# fetch_times.groupby('crawl')['fetch_time'].mean().sort_index().to_csv('C:/Users/Jakob/Downloads/crawl_fetch_times.csv', header=True)
# fetch_times = pd.read_csv('C:/Users/Jakob/Downloads/crawl_fetch_times.csv')
# fetch_times['fetch_time'] = pd.to_datetime(fetch_times.fetch_time)
# all_fetch_dates = pd.to_datetime(fetch_times.fetch_time.dt.date.unique())
# all_fetch_dates = pd.DataFrame(all_fetch_dates, columns=['datadate'])
# all_quarter_closing_dates = df.datadate.value_counts().head(29).sort_index().reset_index()
# crawl_quarter_map
# all_gvkeys = pd.DataFrame(df.gvkey.unique(), columns=['gvkey'])
# all_firm_dates = pd.merge(all_fetch_dates, all_gvkeys, how='cross')
# # # read in chunks
# chunksize = 1e6
# cs_global_sec = pd.read_csv('C:/Users/Jakob/Downloads/compustat-global-daily-securities.csv',
#                             chunksize=chunksize, usecols=['gvkey', 'datadate', 'prccd', 'ajexdi', 'trfd'],
#                             parse_dates=['datadate'])
# chunks = []
# for i, chunk in enumerate(cs_global_sec):
#     print(f'Processing chunk {i}...')
#     chunk.drop_duplicates(subset=['datadate', 'gvkey'], keep='first', inplace=True)
#     chunk.sort_values(['datadate', 'gvkey'], inplace=True)
#     chunk = pd.merge_asof(all_firm_dates, chunk, on='datadate',
#                           direction='backward', tolerance=pd.Timedelta('7 days'), by='gvkey')
#     chunk.dropna(subset=['prccd'], inplace=True)
#     chunks.append(chunk)
#
# cs_na_sec = pd.read_csv('C:/Users/Jakob/Downloads/compustat-na-daily-securities.csv',
#                         chunksize=chunksize, usecols=['gvkey', 'datadate', 'prccd', 'ajexdi', 'trfd'],
#                         parse_dates=['datadate'])
# for i, chunk in enumerate(cs_na_sec):
#     print(f'Processing chunk {i}...')
#     chunk.drop_duplicates(subset=['datadate', 'gvkey'], keep='first', inplace=True)
#     chunk.sort_values(['datadate', 'gvkey'], inplace=True)
#     chunk = pd.merge_asof(all_firm_dates, chunk, on='datadate',
#                           direction='backward', tolerance=pd.Timedelta('7 days'), by='gvkey')
#     chunk.dropna(subset=['prccd'], inplace=True)
#     chunks.append(chunk)
#     break
#
# cs_sec = pd.concat(chunks).drop_duplicates(subset=['datadate', 'gvkey'])
# cs_sec.to_parquet('C:/Users/Jakob/Downloads/compustat-daily-securities_matched_fetch_dates.parquet')

# read compustat cik gvkey link
# cik_gvkeys = []
# for path in [f'C:/Users/Jakob/Downloads/compustat-{}-gvkey-cik.xlsx'.format(x) for x in ['na', 'global']]:
#     cik_gvkey = pd.read_excel('C:/Users/Jakob/Downloads/compustat-na-gvkey-cik.xlsx', usecols=['Global Company Key', 'CIK Number'])
#     cik_gvkey.columns = ['gvkey', 'cik']
#     cik_gvkey.dropna(inplace=True)
#     cik_gvkey.drop_duplicates(inplace=True)
#     cik_gvkey = cik_gvkey.astype(int)
#     cik_gvkeys.append(cik_gvkey)
# cik_gvkey = pd.concat(cik_gvkeys, ignore_index=True).drop_duplicates()
# cik_gvkey.to_csv('C:/Users/Jakob/Downloads/compustat-cik-gvkey.csv', index=False)
cik_gvkey = pd.read_csv('C:/Users/Jakob/Downloads/compustat-cik-gvkey.csv')
df = df.merge(cik_gvkey, on='gvkey', how='left')


# add essential industries flag
with open('/Users/jakob/Downloads/essential_industries.txt', 'r') as f:
    lines = f.readlines()
    lines = [(line.strip().split()[0], ' '.join(line.strip().split()[1:])) for line in lines[1:]]
essential_industries = pd.DataFrame(lines, columns=['naics', 'name'])
df['essential_industry'] = df.naics.astype(str).str[:4].isin(essential_industries.naics.astype(str))
df.groupby('essential_industry').gvkey.nunique()

# fill NA with 0 (i.e. pre-pandemic by definition 0)
llm_vars = [col for col in df.columns if any([col.endswith(x) for x in experiment_labels.values()])]
df[llm_vars] = df[llm_vars].fillna(False)
llm_affected_vars = [col for col in llm_vars if col.startswith('affected')]
df[llm_affected_vars] = df[llm_affected_vars].replace(False, 0)

## export to STATA
df['stata_date'] = df.date.astype(str)
df.drop(columns=['date', 'atq_gr', 'gpq_gr', 'revtq_gr', 'merge_name', 'Country Code', 'Admin1 Code', 'stock_closing', 'log_stock_closing',
                 'Admin2 Code', 'Admin3 Code', 'Admin4 Code', 'RegionName_state']).rename(
    columns={'stata_date': 'date'}).to_stata('C://Users/Jakob/Downloads/reg_df_29-10-24.dta')

df = pd.read_stata('C://Users/Jakob/Downloads/reg_df_29-10-24.dta')
naics_to_nace = compustat[['nace', 'naics']].drop_duplicates().dropna().astype(int)
naics_to_nace.to_csv('C://Users/Jakob/Downloads/naics_to_nace_compustat.csv', index=False)

## heatmap correlations between tags
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df[['hygiene_measures_llama', 'remote_work_llama',
       'supply_chain_issues_llama', 'closure_llama', 'other_llama']].corr()
mask = np.eye(correlation_matrix.shape[0], dtype=bool)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .8})
plt.title('Heatmap of Pairwise Correlations', size=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

df.groupby('gvkey').covid_mention.max().value_counts(normalize=True).sort_index()
df.groupby('gvkey').affected_llama.max().value_counts(normalize=True).sort_index()
df.groupby('gvkey').affected_gpt.max().value_counts(normalize=True).sort_index()

df[df.affected_llama>0].gvkey.nunique()/df.gvkey.nunique()
df[df.affected_gpt>0].gvkey.nunique()/df.gvkey.nunique()

# df = pd.read_stata('C://Users/Jakob/Downloads/reg_df_19-08-24.dta')

# create a table with the number and share of affected firms (in each category)
def create_row_data(column_name):
    value_counts = df.groupby('gvkey')[column_name].max().value_counts().to_dict()
    row_data = [value_counts.get(i, 0) for i in range(4)] + [(df.groupby('gvkey')[column_name].max()>0).sum(), df.gvkey.nunique()]
    return row_data

# Create the rows for the affected_table
llama_row = create_row_data('affected_llama')
gpt_row = create_row_data('affected_gpt')

# Construct the DataFrame
affected_table = pd.DataFrame([llama_row, gpt_row],
                              index=['llama', 'gpt'],
                              columns=['Affected=0', 'Affected=1', 'Affected=2', 'Affected=3', 'Affected $>$ 0', 'Total N'])

# add shares
non_total_cols = [col for col in affected_table.columns if col != 'Total N']
for col in non_total_cols:
    affected_table[col] = affected_table[col] / affected_table['Total N']

affected_table = affected_table[['Affected=0', 'Affected $>$ 0', 'Affected=1', 'Affected=2', 'Affected=3',
       'Total N']].T

print(affected_table*100)

affected_table.to_latex('C://Users/Jakob/Downloads/affected_table.tex',
                        label='tab:affected_table',
                        caption='Share of affected firms in each category',
                        float_format="%.2f")

## add hassan data
# read replication package hassan data
hassan_transcript_scores = pd.read_csv(
    'C:/Users/Jakob/Downloads/Replication Package Pseudo Data/Pseudo-data/Python/Input/transcriptlevel_scores.csv')
hassan_transcript_scores = hassan_transcript_scores[hassan_transcript_scores.disease_string == 'covid']

hassan_transcript_meta = pd.read_csv(
    'C:/Users/Jakob/Downloads/Replication Package Pseudo Data/Pseudo-data/Python/Input/transcriptlevel_metadata.csv')

hassan = hassan_transcript_scores.merge(hassan_transcript_meta, on='transcript_id')
del hassan_transcript_scores, hassan_transcript_meta

hassan['date'] = pd.to_datetime(hassan['dt'])
hassan['date'] = hassan.date.dt.to_period('Q')
hassan = hassan[hassan['disease_string']=='covid']
hassan = hassan[hassan.date.dt.year >= 2020]

hassan.dt.value_counts()

# check that names are the same
test = df[['gvkey', 'conm']].merge(hassan[['gvkey', 'company_name']], on='gvkey', how='inner').drop_duplicates()
print(f'Matched {test.gvkey.nunique()} out of {df.gvkey.nunique()} companies ({test.gvkey.nunique()/df.gvkey.nunique()*100:.2f}%) to Hassan data')

hassan.groupby(['gvkey', 'date']).disease_exposure.max()

len(hassan.drop_duplicates(subset=['transcript_id'])) == len(hassan) # each row is a transcript

hassan_cols = ['disease_exposure', 'disease_risk', 'disease_sentiment', 'negative demand shock',
               'negative supply chain shock', 'negative production and operations shock']
hassan_grouped_mean = hassan.groupby(['gvkey', 'date'])[hassan_cols].mean()
hassan_grouped_mean.rename(columns={'negative demand shock': 'demand',
                                    'negative supply chain shock': 'supply',
                                    'negative production and operations shock': 'production'}, inplace=True)
hassan_grouped_mean.columns = [f'hassan_{col}' for col in hassan_grouped_mean.columns]

df = df.merge(hassan_grouped_mean, on=['gvkey', 'date'], how='left')

gvkeys_in_hassan = set(hassan.gvkey.unique())
df[df.gvkey.isin(gvkeys_in_hassan)].gvkey.nunique()

# fill pre 2020 with 0 for firms in Hassan data
# df.loc[df.gvkey.isin(gvkeys_in_hassan) & (df.date.dt.year < 2020), hassan_grouped_mean.columns] = 0

# for Hassan firms, fill all NA with 0
# df.loc[df.gvkey.isin(gvkeys_in_hassan), hassan_grouped_mean.columns] = df.loc[
#     df.gvkey.isin(gvkeys_in_hassan), hassan_grouped_mean.columns].fillna(0)

corr_columns = ['covid_mention', 'affected_llama', 'affected_gpt',
                'demand_affected_llama', 'supply_affected_llama', 'production_affected_llama']
corrs_hassan = pd.DataFrame(index=hassan_grouped_mean.columns, columns=corr_columns)
for col1 in hassan_grouped_mean.columns:
    for col2 in corr_columns:
        corrs_hassan.loc[col1, col2] = df[col1].corr(df[col2])

## stephany data
stephany = pd.read_csv('/Users/Jakob/Downloads/CoRisk-Index-Data/10x_report_sentences.csv')
stephany = stephany[['cik', 'date', 'report_corona_count', 'report_word_count']].copy()
stephany.cik = stephany.cik.astype(float)
stephany['date'] = pd.to_datetime(stephany['date'])
stephany['date'] = stephany.date.dt.to_period('Q')
stephany = stephany.groupby(['cik', 'date']).sum()
stephany['word_share_corona'] = stephany['report_corona_count'] / stephany['report_word_count']
stephany.drop(columns=['report_corona_count', 'report_word_count'], inplace=True)
stephany['covid_mention'] = stephany['word_share_corona'] > 0
stephany.columns = ['stephany_' + col for col in stephany.columns]
df = df.merge(stephany.reset_index(), on=['cik', 'date'], how='left')

corr_columns = ['covid_mention', 'affected_llama', 'affected_gpt']
corrs_stephany = pd.DataFrame(index=stephany.columns, columns=corr_columns)
for col1 in stephany.columns:
    for col2 in corr_columns:
        corrs_stephany.loc[col1, col2] = df[col1].corr(df[col2])

## brynolfsson
df['remote work_llama'].corr(df.wfh_index*df['C2M_Workplace closing'])

## oxford policy tracker
df['remote work_llama'].corr(df['C2M_Workplace closing'])


# investigate by industry
affectedness_by_industry = df.groupby('nace_2_digit').affected_llama.mean().sort_values(ascending=False).to_frame().join(nace.astype(str).set_index('nace_2_digit'))

llama_tags = [tag + '_llama' for tag in tags]
tag_counts_by_industry = df.groupby('nace_2_digit')[llama_tags].mean().join(nace.astype(str).set_index('nace_2_digit'))
tag_counts_by_industry.sort_values('supply chain issues_llama', ascending=False, inplace=True)

## export Compustat data for Karim
# df = pd.read_stata('C://Users/Jakob/Downloads/reg_df.dta')
#
# cols = ['gvkey', 'date', 'conm', 'loc', 'city', 'addzip', 'Coordinates', 'url', 'nace', 'nace_2_digit',
#         'saleq_usd', 'atq_usd', 'gpq_usd', 'revtq_usd', 'saleq_gr', 'returnq', 'production_neg_sent',
#         'demand_neg_sent', 'supply_neg_sent', 'covid_mention', 'cc_data_available']
#
# df = df[cols].copy()
#
# df.to_csv('C://Users/Jakob/Downloads/compustat_commoncrawl_covid.csv', index=False)

df.nace_section.value_counts(dropna=False)
df.groupby('region').gvkey.nunique()
df.groupby('sub-region').gvkey.nunique()




## panel OLS regression of turnover growth on covid mention
from linearmodels import PanelOLS
from linearmodels.panel import compare

explanatory_var = 'covid_mention'
dep_var = 'saleq_gr'
controls = [] #['l1_log_atq_usd']
lag = 0
reg_df = df[['url', 'date', dep_var, explanatory_var] + controls].copy()
if lag > 0:
    reg_df[f'l{lag}_{explanatory_var}'] = reg_df.groupby('url')[explanatory_var].shift(lag)
    explanatory_var = f'l{lag}_{explanatory_var}'
reg_df.dropna(inplace=True)

# take years 2019 and 2020
reg_df = reg_df[reg_df.date.between('2018Q1', '2022Q4')]

# make panel balanced (by dropping observations with missing values)
# obs_per_period = reg_df.url.value_counts()
# reg_df = reg_df[reg_df.url.isin(obs_per_period[obs_per_period == obs_per_period.max()].index)]

reg_df['date'] = reg_df.date.dt.to_timestamp()
# set date as index
reg_df.set_index(['url', 'date'], inplace=True)

fe_mod = PanelOLS(reg_df[dep_var], reg_df[[explanatory_var] + controls], entity_effects=True, time_effects=True)
fe_res = fe_mod.fit(cov_type='clustered', cluster_entity=True)
print(fe_res)

df.saleq_gr.describe()

df.groupby('date').saleq_gr.mean()
df.city.value_counts().nunique()

# good results: log turnover (level) on l1 or l2 covid mention -> 2% lower turnover - predicts ahead of time for years 2019 + 2020
# bad results: turnover growth on l0 or l1 of covid mention -> 2/6 pct. points lower turnover growth for years 2019 + 2020



explanatory_vars = ['covid_mention', 'production_neg_sent', 'demand_neg_sent', 'supply_neg_sent', 'travel_neg_sent', 'finance_neg_sent']
results = []
for explanatory_var in explanatory_vars:
    reg_df = df[['url', 'date', 'saleq_gr', explanatory_var]].dropna()

    # take years 2019 and 2020
    # reg_df = reg_df[reg_df.date.between('2019Q1', '2021Q4')]

    # make panel balanced
    # obs_per_period = reg_df.url.value_counts()
    # reg_df = reg_df[reg_df.url.isin(obs_per_period[obs_per_period == obs_per_period.max()].index)]

    # set date as index
    reg_df.date = reg_df.date.dt.to_timestamp()
    reg_df.set_index(['url', 'date'], inplace=True)

    # reg_df = reg_df[reg_df.saleq_gr.between(-100, reg_df.saleq_gr.quantile(0.99))]
    reg_df.saleq_gr = reg_df.saleq_gr.clip(lower=-100, upper=reg_df.saleq_gr.quantile(0.




                                                                                      ))

    fe_mod = PanelOLS(reg_df.saleq_gr, reg_df[[explanatory_var]], entity_effects=True, time_effects=True)
    fe_res = fe_mod.fit(cov_type='clustered', cluster_entity=True)
    results.append(fe_res)

print(compare(results).summary)


df.groupby('date').saleq.median().plot()
plt.show()


# to latex
print(fe_res.summary.as_latex())

import pandas as pd
df = pd.read_csv('C://Users/Jakob/Downloads/orbis-2023-05-17/All_addresses/All_addresses.txt',
                 nrows=1000, sep='\t')