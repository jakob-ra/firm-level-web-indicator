import awswrangler as wr
import pandas as pd
from utils import remove_url_prefix
import numpy as np

# df = pd.read_parquet('C:/Users/Jakob/Downloads/res_llm_consolidated.parquet')
# df.columns
df = wr.s3.read_parquet('s3://cc-download-orbis-global/res_llm_consolidated/res_llm_consolidated_full.parquet')
domain_index_cols = ['url_host_registered_domain', 'date']
df[domain_index_cols] = df[domain_index_cols].astype('category')
df = df.set_index(domain_index_cols).sort_index()

affectedness_categories = ['production_affected', 'demand_affected', 'supply_affected']
tags = ['hygiene measures', 'remote work', 'supply chain issues', 'closure', 'financial impact', 'travel restrictions']
tags = [x.replace(' ', '_') for x in tags]

# export results for Swiss survey firms for Martin
# swiss_survey_urls = pd.read_excel('/Users/Jakob/Downloads/kof_domains_ids_survey.xlsx',
#                                   skiprows=1,
#                                   names=['drop', 'id', 'url'])
# swiss_survey_urls.drop(columns='drop', inplace=True)
# res_swiss = res[res.url_host_registered_domain.isin(swiss_survey_urls.url)]
# res_swiss.to_csv('/Users/Jakob/Downloads/res_swiss.csv', index=False)
#
# res_swiss.url_host_registered_domain.nunique()
# swiss_survey_urls.url.nunique()

# add dataprovider high heartbeat dummy
dataprovider = wr.s3.read_csv('s3://cc-download-orbis-global/dataprovider/All companies for profit high heartbeat all languages.csv', usecols=['Hostname'])
dataprovider['url_host_registered_domain'] = dataprovider['Hostname'].apply(remove_url_prefix)
dataprovider = dataprovider.drop(columns=['Hostname']).drop_duplicates().set_index('url_host_registered_domain').sort_index()
dataprovider['dataprovider_high_heartbeat'] = True
df = df.join(dataprovider, on='url_host_registered_domain', how='left')
df['dataprovider_high_heartbeat'] = df['dataprovider_high_heartbeat'].fillna(False)
del dataprovider

# content digest heartbeat
cdh = wr.s3.read_parquet('s3://cc-download-orbis-global/res_llm_consolidated/res_llm_consolidated_content_digest_heartbeat.parquet')
df = df.join(cdh, on='url_host_registered_domain', how='left')
df['content_digest_heartbeat'] = df['content_digest_heartbeat'].astype(np.float16)

# match with Orbis
orbis = pd.read_parquet('C:/Users/Jakob/Downloads/swiss_covid_orbis_global_full.parquet')
orbis_dynamic = pd.read_parquet('C:/Users/Jakob/Downloads/swiss_covid_orbis_global_dynamic_full.parquet')
orbis_dynamic.rename(columns={'closing_date': 'year'}, inplace=True)

# for each domain keep only bvdids w highest number of employees and highest turnover and prefer available nace code
# orbis.bvdid.nunique() = 12 mil
keep_bvdids = orbis[['bvdid', 'nace_2_digit', 'url_host_registered_domain']].merge(orbis_dynamic, on='bvdid', how='left').sort_values(['operating_revenue_turnover', 'total_assets', 'number_of_employees'], ascending=False).drop_duplicates(subset=['url_host_registered_domain'], keep='first').bvdid.unique()
orbis = orbis[orbis.bvdid.isin(keep_bvdids)].copy(deep=True)
orbis_dynamic = orbis_dynamic[orbis_dynamic.bvdid.isin(keep_bvdids)].copy(deep=True)
orbis = orbis.set_index('url_host_registered_domain').sort_index()

# group multiple domains for same bvdid
df = df.join(orbis[['bvdid']], on='url_host_registered_domain', how='inner') # inner: keep only firms with bvdid (in this dataset=has nace code, location)
orbis = orbis.reset_index(drop=True).set_index('bvdid')
df = df.reset_index().set_index(['bvdid', 'date']).sort_index()
agg_dict = {'affected': 'max', 'covid_mention': 'max'}
agg_dict.update({cat: 'max' for cat in affectedness_categories})
agg_dict.update({tag: 'max' for tag in tags})
agg_dict.update({'dataprovider_high_heartbeat': 'max', 'cc_data_available': 'max', 'content_digest_heartbeat': 'max'})
# agg_dict.update({'url_host_registered_domain': lambda x: list(x)})
df = df.groupby(level=[0,1], observed=True).agg(agg_dict) # this takes a while

# merge with orbis static
df = df.join(orbis.drop(columns=['name_internat']), on='bvdid', how='left', validate='m:1')
del orbis

# merge with orbis_dynamic
df.reset_index(inplace=True)
df.date = pd.to_datetime(df.date)
df = df.merge(orbis_dynamic, left_on=['bvdid', df.date.dt.year], right_on=['bvdid', 'year'], how='left', validate='m:1', indicator=True)
print('Results of merge with orbis_dynamic:\n', df['_merge'].value_counts())
df.drop(columns='_merge', inplace=True)
del orbis_dynamic
df.year = df.year.astype('category')
df.set_index(['bvdid', 'date'], inplace=True)

# memory optimization
float_cols = ['number_of_employees', 'operating_revenue_turnover', 'total_assets']
df[float_cols] = df[float_cols].astype('float32')

# memory usage
memory_usage = df.memory_usage(deep=True)
memory_usage.div(1e9).round(2)

df.to_parquet('C:/Users/Jakob/Downloads/cc_res_llm_deduplicated_bvdid_merged_orbis_orbis_dynamic_25_11.parquet')
wr.s3.to_parquet(df, 's3://cc-download-orbis-global/res_llm_consolidated/cc_res_llm_deduplicated_bvdid_merged_orbis_orbis_dynamic_25_11.parquet')