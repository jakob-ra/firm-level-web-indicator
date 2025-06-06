import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
import plotting
import seaborn as sns
from utils import get_oxford_policy_tracker, float_cols_to_float64
import country_converter as coco
import geopandas as gpd
import json
from matplotlib.ticker import PercentFormatter

# tag mapping table
with open('tag_mapping.json', 'r') as f:
    tag_mapping = json.load(f)

for tag in tag_mapping.keys():
    print(', '.join(tag_mapping[tag]))

affectedness_categories = ['production_affected', 'demand_affected', 'supply_affected']
tags = ['hygiene measures', 'remote work', 'supply chain issues', 'closure', 'financial impact', 'travel restrictions']
tags = [x.replace(' ', '_') for x in tags]
wai_labels = {'affected': 'Mean affectedness (scale 0-3)',
              'affected_min_1': 'Share firms affectedness ($>$0)',
              'affected_min_2': 'Share firms affectedness ($>=$2)',
              'affected_3': 'Share of severely affected firms'}

relevant_nace_sections = ['Accommodation and food service activities', 'Arts, entertainment and recreation',
                          'Construction', 'Education', 'Financial and insurance activities',
                          'Human health and social work activities', 'Information and communication',
                          'Manufacturing', 'Transporting and storage',
                          'Wholesale and retail trade; repair of motor vehicles and motorcycles']
relevant_nace_2_digit = ['Retail trade, except of motor vehicles and motorcycles',
                         # 'Wholesale trade, except of motor vehicles and motorcycles',
                         'Human health activities', 'Food and beverage service activities', 'Education',
                         'Construction of buildings',
                         # 'Computer programming, consultancy and related activities',
                         # 'Land transport and transport via pipelines',
                         'Accommodation', # 'Manufacture of food products',
                         'Travel agency, tour operator and other reservation service and related activities',
                         # 'Manufacture of electrical equipment',
                         # 'Manufacture of chemicals and chemical products',
                         # 'Telecommunications',
                         'Creative, arts and entertainment activities',
                         # 'Manufacture of motor vehicles, trailers and semi-trailers',
                         'Air transport', ]

df = pd.read_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/cc_res_llm_deduplicated_bvdid_merged_orbis_orbis_dynamic_25_11.parquet')

# set dtype of multi-index
df.reset_index(inplace=True)
df.bvdid = df.bvdid.astype('category')
df.date = df.date.astype('category')

# add full employee data
df.drop(columns=['number_of_employees'], inplace=True)
emp = pd.read_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/orbis_global_bvdids_employees.parquet')
emp.drop(columns=['closing_date'], inplace=True)
emp.bvdid = emp.bvdid.astype('category')
df = df.merge(emp, on='bvdid', how='left')
del emp

# memory optimization
df.year = df.year.astype('category')
df.nace_rev_2_core_code_4_digits = df.nace_rev_2_core_code_4_digits.astype('category')
# memory_usage = df.memory_usage(deep=True)
# memory_usage.div(1e9).round(3)

# export unique bvdids
# unique_bvdids = df.bvdid.drop_duplicates()
# unique_bvdids.to_csv('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/global_sample_bvdid_list.csv', index=, header=False)

## fix geocoding for map plots
# fix Chinese territories
df.loc[df.country == 'Hong Kong SAR, China', 'country'] = 'China'
df.loc[df.country == 'Taiwan, China', 'country'] = 'China'

# fix administrative regions
df['region_level_1'] = df.region_in_country.str.split('|').str[0]
to_fix = ['Russian Federation', 'Hungary', 'Japan', 'Vietnam', 'China', 'Norway', 'Romania']
df.loc[df.country.isin(to_fix), 'region_level_1'] = df.loc[df.country.isin(to_fix), 'region_in_country'].str.split('|').str[1]
df['region_level_1'] = df['region_level_1'].astype('category')

# convert both columns back to category
df[['country', 'region_level_1']] = df[['country', 'region_level_1']].astype('category')

country_counts = df.groupby('country', observed=True).bvdid.nunique().sort_values(ascending=False).reset_index()

# remove countries from analysis according to criteria in table A1
country_firms_table = pd.read_csv('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/tables/country_firms_table.csv')
use_countries = country_firms_table.head(100).copy()
use_countries = use_countries[(use_countries.num_firms_analyzed_per_m_pop>=200) & (use_countries.share_firms_analyzed_total_except_micro_table>=0.2)].country.to_list()
drop_countries = ['Hong Kong SAR, China',  'Luxembourg', 'Montenegro', 'Bermuda', 'Liechtenstein', 'Slovenia', 'Republic of Moldova']
use_countries = [c for c in use_countries if c not in drop_countries]

# oecd share of people using internet
oecd = pd.read_excel('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/countries_internet_usage.xlsx')
oecd.rename(columns={'Code': 'country_iso3',
                     'Individuals using the Internet (% of population)': 'share_people_using_internet',
                     'Entity': 'country'}, inplace=True)
remove_threshold = oecd[oecd.country=='European Union (27)'].share_people_using_internet.iloc[0]
remove_countries = oecd[oecd.share_people_using_internet<remove_threshold].country_iso3
use_countries = [c for c in use_countries if coco.convert(names=c, to='ISO3') not in remove_countries.to_list()]

df = df[df.country.isin(use_countries)].copy()

date_counts = df.date.value_counts().sort_index()
dates = date_counts.index.to_series()

covid_mentioning_urls = list(df[df.covid_mention].bvdid.unique())

def format_date_axis(ax):
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    return None

# content digest heartbeat histogram
firms_cdh = df.groupby('bvdid').content_digest_heartbeat.first()
plt.hist(firms_cdh, weights=np.ones(len(firms_cdh)) / len(firms_cdh), bins=20, color='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('Content digest heartbeat')
plt.ylabel('Share of firms')
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/content_digest_heartbeat_hist.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/content_digest_heartbeat_hist.png', dpi=300, bbox_inches='tight')
plt.show()
firms_cdh.describe()

# number of firms in each crawl
num_of_firms_crawled_by_date = df[df.affected.notna()].reset_index().groupby('date').bvdid.nunique().sort_index()
num_of_firms_crawled_by_date.plot()
plt.show()

# number of firms crawled for each industry
industry_firm_count = df[df.affected.notna()].reset_index().groupby('nace_section', observed=True).bvdid.nunique().sort_values(ascending=False)
plotting.plot_barh(industry_firm_count, xlabel='Number of firms', extend_x_axis=0.2, label_fmt='{:,.0f}')

# number of firms crawled for each country
plotting.plot_barh(country_counts.set_index('country').head(60).squeeze(), xlabel='Number of firms', ylabel='Country',
                   extend_x_axis=0.1)

# plot number of domains with at least one subpage mentioning covid over time
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
to_plot = df.groupby('date', observed=True).affected.count()
to_plot.index = pd.to_datetime(to_plot.index)
to_plot.plot(ax=ax1, color='blue')
to_plot = df.groupby('date').affected.count().div(
    df.groupby('date').size()) * 100
to_plot.index = pd.to_datetime(to_plot.index)
to_plot.plot(ax=ax2, color='orange')
ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
format_date_axis(ax1)
ax1.set_ylabel('Number of domains mentioning covid', color='blue')
ax2.set_ylabel('Share of domains mentioning covid (%)', color='orange')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='orange')
ax1.set_xlabel('')
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//num_and_share_domains_mentioning_covid_llm.pdf', dpi=300,
            bbox_inches='tight')
plt.show()

## share of domains mentioning covid, all vs high heartbeat vs large
fig, ax = plt.subplots(figsize=(10, 5))
to_plot = df.groupby('date').covid_mention.sum().div(
    df.groupby('date').cc_data_available.sum()) * 100
to_plot.plot(ax=ax, label='All')
to_plot = df[df.dataprovider_high_heartbeat].groupby('date').covid_mention.sum().div(
    df[df.dataprovider_high_heartbeat].groupby('date').cc_data_available.sum()) * 100
to_plot.plot(ax=ax, label='High heartbeat')
max_employees = df.groupby(df.bvdid).number_of_employees.max()
min_50_empl = df.bvdid.isin(max_employees[max_employees >= 50].index)
to_plot = df[min_50_empl].groupby('date').covid_mention.sum().div(
    df[min_50_empl].groupby('date').cc_data_available.sum()) * 100
to_plot.plot(ax=ax, label='50+ employees')
to_plot = df[df.content_digest_heartbeat==1].groupby('date').covid_mention.sum().div(
    df[df.content_digest_heartbeat==1].groupby('date').cc_data_available.sum()) * 100
to_plot.plot(ax=ax, label='High heartbeat (content digest)')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
format_date_axis(ax)
ax.set_ylabel('Share of domains mentioning covid (%)')
ax.set_xlabel('')
ax.legend()
plt.savefig(
    '/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//share_covid_mentioning_firms_all_vs_high_heartbeat_vs_50_empl_6_months_expiry_llm.pdf',
    dpi=300, bbox_inches='tight')
plt.show()

## mean affectedness, all vs high heartbeat vs large
fig, ax = plt.subplots(figsize=(10, 5))
to_plot = df.groupby('date').affected.mean()
to_plot.plot(ax=ax, label='All')
to_plot = df[df.dataprovider_high_heartbeat].groupby('date').affected.mean()
to_plot.plot(ax=ax, label='High heartbeat')
to_plot = df[min_50_empl].groupby('date').affected.mean()
to_plot.plot(ax=ax, label='50+ employees')
to_plot = df[df.content_digest_heartbeat==1].groupby('date').affected.mean()
to_plot.plot(ax=ax, label='High heartbeat (content digest)')
format_date_axis(ax)
ax.set_ylabel('Mean affectedness')
ax.set_xlabel('')
ax.legend()
plt.savefig(
    '/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//mean_affected_all_vs_high_heartbeat_vs_50_empl_llm.pdf',
    dpi=300, bbox_inches='tight')
plt.show()


# calculate shares by industry over time
industry_type = 'nace_section'  # 'nace_2_digit'
# df[df.index.get_level_values('url_host_registered_domain').isin(covid_mentioning_urls)]
# df[df.content_digest_heartbeat==1]
mean_affected_by_industry = df[df.dataprovider_high_heartbeat].groupby(['date', industry_type]).affected.mean().reset_index()

to_plot = mean_affected_by_industry.pivot(index='date', columns=industry_type, values='affected')
# top 10 affected industries
# to_plot = to_plot[to_plot.max().sort_values(ascending=False).head(10).index]
# top 10 biggest industries
# to_plot = to_plot[total_firms_by_industry.sort_values('total_firms', ascending=False).head(10)[industry_type]]
# pre-defined industries
if industry_type == 'nace_section':
    to_plot = to_plot[relevant_nace_sections]
elif industry_type == 'nace_2_digit':
    to_plot = to_plot[relevant_nace_2_digit]
ax = to_plot.plot()
format_date_axis(ax)
# order legend by max value
handles, labels = ax.get_legend_handles_labels()
handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: to_plot[t[1]].max(), reverse=True))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
leg = ax.legend(handles, labels, loc='upper left', title='Industry', bbox_to_anchor=(1, 0.8))
leg._legend_box.align = 'left'
ax.set_xlabel('')
ax.set_ylabel(f'Mean affected')
plt.gcf().set_size_inches(10, 5)
plt.tight_layout()
plt.savefig(
    f'/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//mean_affected_by_industry_llm.pdf',
    dpi=300, bbox_inches='tight')
plt.show()

# mean affectedness by country
mean_affected_by_country = df.groupby(['date', 'country'])['affected'].apply(lambda x: x.mean(skipna=True))
to_plot = mean_affected_by_country.reset_index().pivot(index='date', columns='country', values='affected')
# top 10 biggest countries
to_plot = to_plot[country_counts.head(10).country]
# top 10 affected countries
# to_plot = to_plot[to_plot.max().sort_values(ascending=False).head(10).index]
ax = to_plot.plot()
format_date_axis(ax)
# order legend by max value
handles, labels = ax.get_legend_handles_labels()
handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: to_plot[t[1]].max(), reverse=True))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
leg = ax.legend(handles, labels, loc='upper left', title='Industry', bbox_to_anchor=(1, 0.8))
leg._legend_box.align = 'left'
ax.set_xlabel('')
ax.set_ylabel(f'Mean affectedness')
plt.gcf().set_size_inches(10, 5)
plt.tight_layout()
plt.savefig(
    f'/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/mean_affected_by_country_llm.pdf',
    dpi=300, bbox_inches='tight')
plt.show()

## oxford policy tracker web indicators - important: C2E_Workplace closing, C6E_Stay at home requirements
oxford_policy_tracker = get_oxford_policy_tracker(aggregation_period='M', extended_cols=True)
oxford_policy_tracker.rename(columns={'CountryCode': 'country', 'Date': 'month'}, inplace=True)
oxford_policy_tracker.drop(columns=['StringencyIndex_Average_ForDisplay'], inplace=True)
oxford_policy_tracker_cols = [col for col in oxford_policy_tracker.columns if col not in ['loc', 'date']]
# center and normalize all index columns
index_cols = ['StringencyIndex_Average', 'GovernmentResponseIndex_Average', 'ContainmentHealthIndex_Average',
              'EconomicSupportIndex']
# oxford_policy_tracker[index_cols] = (oxford_policy_tracker[index_cols] - oxford_policy_tracker[
#     index_cols].mean()) / oxford_policy_tracker[index_cols].std() # normalize index columns
# ## convert fiscal measures to % of GDP
# path = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv'
# import requests
# from zipfile import ZipFile
# from io import BytesIO
# with ZipFile(BytesIO(requests.get(path).content)) as zf:
#     with zf.open('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5871885.csv') as f:
#         gdp_df = pd.read_csv(f, header=2)
# gdp_df.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code'], inplace=True)
# gdp_df = pd.wide_to_long(gdp_df, "", i="Country Code", j="year")
# gdp_df.columns = ['', 'gdp']
# gdp_df = gdp_df[['gdp']]
# # ffill missing values
# gdp_df['gdp'] = gdp_df.groupby('Country Code').gdp.ffill()
# gdp_df.reset_index(inplace=True)
# gdp_df = gdp_df[gdp_df.year == 2020].drop(columns=['year'])
# oxford_policy_tracker = oxford_policy_tracker.merge(gdp_df, left_on='country', right_on='Country Code', how='left').drop(columns=['Country Code'])
# oxford_policy_tracker['fiscal_measures_pct_gdp'] = oxford_policy_tracker['E3_Fiscal measures'] / oxford_policy_tracker['gdp'] * 100
# oxford_policy_tracker.sort_values('gdp', ascending=False).head(20)
# oxford_policy_tracker_cols += ['fiscal_measures_pct_gdp']
# round all ordinal scale variables to integers
scale_vars = ['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport',
              'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief']
oxford_policy_tracker[scale_vars] = oxford_policy_tracker[scale_vars].round(0)

# get dummies for affected
df = df.join(pd.get_dummies(df.affected.astype(pd.Int64Dtype()), drop_first=True, prefix='affected', dtype='boolean'))
affected_dummy_cols = ['affected_1', 'affected_2', 'affected_3']
df.loc[~df.cc_data_available, affected_dummy_cols] = pd.NA # set affected dummies to NA if no cc data available
df['affected_min_1'] = df.affected>=1
df['affected_min_2'] = df.affected>=2
affected_min_dummy_cols = ['affected_min_1', 'affected_min_2']
df[affected_min_dummy_cols] = df[affected_min_dummy_cols].astype('boolean')
df.loc[~df.cc_data_available, affected_min_dummy_cols] = pd.NA # set affected dummies to NA if no cc data available

# calculate aggregations by country and crawl
wai_cols = ['covid_mention', 'affected'] + affectedness_categories + tags + affected_dummy_cols + affected_min_dummy_cols
agg_dict = {x: 'mean' for x in wai_cols}
agg_dict.update({'cc_data_available': 'sum'})
affected_by_country = df.groupby(['date', 'country'], observed=True).agg(agg_dict)
# add an entry for Europe
df['world_region'] = np.nan
country_counts['region'] = coco.convert(names=country_counts.country, to='UNregion')
european_countries = country_counts[country_counts.region.str.contains('Europe').fillna(False)].country.to_list()
european_countries = [c for c in european_countries if c not in ['Russian Federation']]
df.loc[df.country.isin(european_countries), 'world_region'] = 'Europe'
affected_by_world_region = df.groupby(['date', 'world_region'], observed=True).agg(agg_dict)
affected_by_world_region.index.set_names(['date', 'country'], inplace=True)
affected_by_country = pd.concat([affected_by_country, affected_by_world_region])
df.world_region = df.world_region.astype('category')

world_region_counts = df.groupby('world_region', observed=True).bvdid.nunique().sort_values(ascending=False).reset_index()
world_region_counts.rename(columns={'world_region': 'country'}, inplace=True)
country_counts = pd.concat([country_counts, world_region_counts])

# aggregate oxford policy tracker for Europe (mean over countries)
oxford_plot_cols = ['StringencyIndex_Average', 'C2M_Workplace closing', 'C6M_Stay at home requirements']
oxford_policy_tracker['world_region'] = np.nan
oxford_policy_tracker.loc[oxford_policy_tracker.country.isin(coco.convert(names=european_countries, to='ISO3')), 'world_region'] = 'Europe'
oxford_policy_tracker_europe = oxford_policy_tracker.groupby(['world_region', 'month'])[oxford_plot_cols].mean().reset_index()
oxford_policy_tracker_europe.rename(columns={'world_region': 'country'}, inplace=True)
oxford_policy_tracker_europe[['C2M_Workplace closing', 'C6M_Stay at home requirements']] = oxford_policy_tracker_europe[['C2M_Workplace closing', 'C6M_Stay at home requirements']].round(0).astype(int)
oxford_policy_tracker = pd.concat([oxford_policy_tracker, oxford_policy_tracker_europe])

# calculate aggregations by state and crawl
agg_dict = {x: 'mean' for x in wai_cols}
agg_dict.update({'cc_data_available': 'sum'})
affected_by_state = df[df.country=='United States of America'].reset_index().groupby(['date', 'region_level_1'], observed=True).agg(agg_dict)
firm_count_by_state = df[df.country=='United States of America'].groupby(['region_level_1', 'bvdid'], observed=True).cc_data_available.max()
firm_count_by_state = firm_count_by_state.reset_index().groupby('region_level_1', observed=True).cc_data_available.sum().sort_values(ascending=False)
from utils import get_oxford_policy_tracker_regional
oxford_policy_tracker_US = get_oxford_policy_tracker_regional('United States', aggregation_period='M',
                                                              extended_cols=True)
oxford_policy_tracker_US.rename(columns={'RegionName': 'region_level_1', 'Date': 'month'}, inplace=True)

# custom nace aggregation
relevant_nace_sections = ['Accommodation and food service activities', 'Arts, entertainment and recreation',
                          'Construction', 'Education', 'Financial and insurance activities',
                          'Human health and social work activities', 'Information and communication',
                          'Manufacturing', 'Transporting and storage',
                          'Wholesale and retail trade; repair of motor vehicles and motorcycles']
relevant_custom_nace_sections = ['Accommodation', 'Food and beverage service activities',
                                 'Arts, entertainment and recreation', 'Construction', 'Education',
                                 'Services', 'Human health and social work activities', 'Manufacturing',
                                 'Transporting and storage', 'Wholesale trade', 'Retail trade', 'Utilities and Environmental Services']
df['nace_custom'] = df['nace_section']
new_categories = ['Wholesale trade', 'Retail trade', 'Food and beverage service activities',
                  'Accommodation', 'Services', 'Utilities and Environmental Services']
df['nace_custom'] = df['nace_custom'].cat.add_categories(new_categories)
df.loc[df.nace_2_digit=='Wholesale trade, except of motor vehicles and motorcycles', 'nace_custom'] = 'Wholesale trade'
df.loc[df.nace_2_digit=='Retail trade, except of motor vehicles and motorcycles', 'nace_custom'] = 'Retail trade'
df.loc[df.nace_2_digit=='Food and beverage service activities', 'nace_custom'] = 'Food and beverage service activities'
df.loc[df.nace_2_digit=='Accommodation', 'nace_custom'] = 'Accommodation'
# df.loc[df.nace_rev_2_core_code_4_digits.str[:3].isin(['491', '511', '493', '501', '503']), 'nace_custom'] = 'Passenger transport'
# df.loc[df.nace_custom=='Transporting and storage', 'nace_custom'] = 'Other transport and storage'


# Services: section  J through N plus S (58 through 83 + 94 through 96)
services_nace_sections = ['Information and communication', 'Financial and insurance activities', 'Real estate activities',
                    'Professional, scientific and technical activities', 'Administrative and support service activities', 'Other services activities']
df.loc[df.nace_section.isin(services_nace_sections), 'nace_custom'] = 'Services'

# Utilities and Environmental Services: section D, E
utilities_nace_sections = ['Electricity, gas, steam and air conditioning supply', 'Water supply; sewerage; waste managment and remediation activities']
df.loc[df.nace_section.isin(utilities_nace_sections), 'nace_custom'] = 'Utilities and Environmental Services'

# aggregations by firm, max over time (careful with dummy aggregation)
agg_dict = {x: 'max' for x in ['covid_mention', 'affected'] + affectedness_categories + tags + affected_min_dummy_cols}
agg_dict.update({'city': 'first', 'region_level_1': 'first', 'country': 'first', 'nace_section': 'first', 'nace_custom': 'first', 'cc_data_available': 'max', 'number_of_employees': 'max'})
affected_max_by_firm = df.groupby('bvdid', observed=True).agg(agg_dict)
affected_max_by_firm = affected_max_by_firm.join(pd.get_dummies(affected_max_by_firm.affected.astype(pd.Int64Dtype()), drop_first=True, prefix='affected', dtype='boolean'))
# check results
affected_max_by_firm.affected.value_counts(dropna=False, normalize=True).sort_index()
affected_max_by_firm.affected_min_1.value_counts(dropna=False, normalize=True).sort_index()


## max affectedness by industry weighted by num employees
affected_by_industry_max_weighted = affected_max_by_firm.copy()
affected_by_industry_max_weighted[wai_cols] = affected_by_industry_max_weighted[wai_cols].multiply(affected_by_industry_max_weighted['number_of_employees'], axis=0)
affected_by_industry_max_weighted['employee_data_available'] = affected_by_industry_max_weighted['number_of_employees'].notna()
agg_dict = {x: 'sum' for x in wai_cols}
agg_dict.update({'cc_data_available': 'sum', 'number_of_employees': 'sum', 'employee_data_available': 'sum'})
affected_by_industry_max_weighted = affected_by_industry_max_weighted.groupby('nace_custom', observed=True).agg(agg_dict)
affected_by_industry_max_weighted[wai_cols] = affected_by_industry_max_weighted[wai_cols].div(affected_by_industry_max_weighted['number_of_employees'], axis=0)
# naics_to_nace = pd.read_csv('C://Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/naics_to_nace_compustat.csv')
# essential_industries.naics = essential_industries.naics.astype(int)
# essential_industries = essential_industries.merge(naics_to_nace, on ='naics', how='left')
# essential_industries.nace = essential_industries.nace.fillna(essential_industries.naics.map(naics_to_nace.set_index('naics').nace.astype(str).str[:4].astype(int)))
# essential_industries.nace.isna().sum()
# essential_nace = [1-11]


to_plot = affected_by_industry_max_weighted.affected_3.loc[relevant_custom_nace_sections].sort_values(ascending=False)*100
sector_name_dict = {'Arts, entertainment and recreation': 'Arts, Entertainment & Recreation',
                                           'Human health and social work activities': 'Human Health & Social Work Activities',
                                           'Retail trade': 'Retail Trade', 'Wholesale trade': 'Wholesale Trade',
                                           'Transporting and storage': 'Transporting & Storage',
                                           'Food and beverage service activities': 'Food & Beverage Service Activities',
                                           'Services': 'Other Services', 'Utilities and Environmental Services': 'Energy Utilities & Environmental Services'}
to_plot.index = to_plot.reset_index().nace_custom.replace(sector_name_dict)
# plotting.plot_barh(to_plot, color='black',
#                    xlabel='Share of severely affected firms (weighted by number of employees)', extend_x_axis=0.1, label_fmt='%.0f%%')
# affected_by_industry_max_weighted.affected_3.loc[relevant_nace_sections].sort_values(ascending=True).plot(kind='barh')
# plt.xlabel('Share of severely affected firms, weighted by number of employees')
# plt.ylabel('Industry')
# plt.show()
label_fmt = '%.0f%%'
to_plot.plot(kind='barh', figsize=(7, 5), color='black')
plt.xlabel('Share of severely affected firms (weighted by number of employees)')
plt.ylabel('')
ax = plt.gca()
ax.bar_label(ax.containers[0], fmt=label_fmt, label_type='edge', padding=3)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter(label_fmt))
ax.set_xlim(right=to_plot.max() + to_plot.max()*0.1)
plt.gca().invert_yaxis()
ax.yaxis.set_minor_locator(plt.NullLocator())
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/weighted_share_severely_affected_firms_by_industry_llm_12_12_24.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/weighted_share_severely_affected_firms_by_industry_llm_12_12_24.png', dpi=300, bbox_inches='tight')
plt.show()

affected_by_industry_max_weighted.employee_data_available.sum()/len(affected_max_by_firm)

# employee weighted affectedness over time
# def weighted_avg(group, value_col, weight_col):
#     return (group[value_col] * group[weight_col]).sum() / group[weight_col].sum()
#
# affected_by_country_weighted = df.groupby(['date', 'country'], observed=True).apply(lambda x: weighted_avg(x, 'affected_3', 'number_of_employees'))

## WAI time series plots
def mix_colors(cf, cb): # cf = foreground color, cb = background color
    a = cb[-1] + cf[-1] - cb[-1] * cf[-1] # fixed alpha calculation
    r = (cf[0] * cf[-1] + cb[0] * cb[-1] * (1 - cf[-1])) / a
    g = (cf[1] * cf[-1] + cb[1] * cb[-1] * (1 - cf[-1])) / a
    b = (cf[2] * cf[-1] + cb[2] * cb[-1] * (1 - cf[-1])) / a
    return [r,g,b,a]

def cross_correlations(s1,s2,lags=range(-5,5)):
    return {lag: s1.corr(s2.shift(lag)) for lag in lags}

def plot_affectedness_policy_tracker(ax, country, wai_data, policy_data, title=False, percent=True,
                                     max_affectedness=None, lw=2, country_counts=None, show_axes=False,
                                     return_twin_axis=False, labels=True, wai_label='Mean affected'):
    if percent:
        fmt = '{x:.0%}'
    else:
        fmt = '{x:,.2f}'
    # Affectedness indicator
    ax1 = ax
    line1 = ax1.plot(wai_data.index, wai_data.values, linewidth=lw,
                     label='Mean affectedness', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    if max_affectedness is not None:
        ax1.set_ylim(0, max_affectedness)

    if show_axes:
        ax1.tick_params(axis='y', labelcolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

    # remove border
    if not show_axes:
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)

    # Government stringency index
    ax2 = ax1.twinx()
    line2 = ax2.plot(policy_data['month'].dt.to_timestamp(), policy_data['StringencyIndex_Average'],
                     label='Stringency Index (OxCGRT)', linewidth=lw, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    ax2.set_ylim(0, 100)

    if show_axes:
        ax2.tick_params(axis='y', labelcolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])

    ax2.spines['top'].set_visible(False)
    if not show_axes:
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

    # axvspan workplace closing
    for j, row in policy_data[policy_data['C2M_Workplace closing'] >= 2].iterrows():
        d = row['month'].to_timestamp()
        axvspan1 = ax1.axvspan(d, d + pd.DateOffset(months=1), color='red', alpha=0.15,
                           label='Workplace closing requirement')

    # axvspan stay at home
    for j, row in policy_data[policy_data['C6M_Stay at home requirements'] >= 2].iterrows():
        d = row['month'].to_timestamp()
        axvspan2 = ax1.axvspan(d, d + pd.DateOffset(months=1), color='yellow', alpha=0.15,
                           label='Stay at home requirement')

    # put correlation of stringency and affectedness in top right corner
    wai_data = wai_data.to_frame().reset_index()
    wai_data['date'] = pd.to_datetime(wai_data.date.dt.strftime('%Y-%m'))
    wai_data = wai_data.set_index('date').squeeze()
    s1 = policy_data.set_index(policy_data['month'].dt.to_timestamp())['StringencyIndex_Average']
    s2 = wai_data
    corr_stringency_affectedness = s1.corr(s2)
    fd_corr_stringency_affectedness = s1.diff().corr(s2.diff())
    ccs = cross_correlations(s1, s2)
    fd_ccs = cross_correlations(s1.diff(), s2.diff())
    ax.text(0.77, 0.9, 'r={:.2f}'.format(fd_corr_stringency_affectedness), transform=ax.transAxes)

    if country_counts is not None:
        ax.text(0.77, 0.8, f'n={country_counts[country_counts.country==country].bvdid.iloc[0]:,.0f}',
                transform=ax.transAxes)
    # turn off top ticks
    ax.tick_params(axis='x', top=False, which='both')

    # Set title for each subplot
    if title==True:
        ax.set_title(f'{country}')

    # add legend label for mixed color
    axvspan_color_mix = mix_colors(mpl.colors.to_rgba('red', alpha=0.15), mpl.colors.to_rgba('yellow', alpha=0.15))
    if 'd' in locals():
        axvspan3 = ax1.axvspan(d, d, color=axvspan_color_mix,
                               label='Workplace closing & stay at home requirement')

    # remove grid, axis labels, and axis ticks for all subplots
    if not show_axes:
        for axis in [ax1, ax2]:
            axis.set_yticks([])
        ax.set_ylabel('')
    else:
        if labels:
            ax1.set_ylabel(wai_label, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax2.set_ylabel('OxCGRT', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        # formatter for ax1
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(fmt))

    try:
        legends = [line1[0], line2[0], axvspan1, axvspan2, axvspan3]
    except:
        legends = [line1[0], line2[0]]

    if return_twin_axis:
        return legends, ax2
    else:
        return legends

# test plot country
country = 'United States of America'
to_plot = affected_by_country.affected_3.rename('affected')
max_affectedness = to_plot.loc[:,country].max()
fig, ax = plt.subplots(figsize=(10, 5))
legends = plot_affectedness_policy_tracker(ax, country, affected_by_country.affected_3.loc[:, country],
                                           oxford_policy_tracker[oxford_policy_tracker.country == coco.convert(country, to='ISO3')], max_affectedness * 1.2,
                                           country_counts=country_counts,
                                           show_axes=True)
fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(2, 0.75))
ax.set_title('United States of America')
plt.tight_layout()
plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//mean_affected_policy_tracker_share_y_corr_2_USA_llm.pdf', dpi=300,
            bbox_inches='tight')
plt.show()

# test plot state
state = 'Michigan'
max_affectedness = affected_by_state.affected_3.loc[:, state].max()
to_plot = affected_by_state.affected_3.rename('affected')
fig, ax = plt.subplots(figsize=(10, 5))
legends = plot_affectedness_policy_tracker(ax, state, to_plot.loc[:, state],
                                                 oxford_policy_tracker_US[oxford_policy_tracker_US.region_level_1==state], max_affectedness*1.2, country_counts=firm_count_by_state.to_frame().reset_index().rename(columns={'region_level_1': 'country', 'cc_data_available': 'bvdid'}),
                                                 show_axes=True)
fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.tight_layout()
plt.show()


# plot tags over time next to policy tracker
def plot_tags_policy_tracker(ax, country, wai_data, policy_data, title=False, percent=True,
                                     max_affectedness=None, lw=2, country_counts=None, show_axes=False,
                                     return_twin_axis=False, labels=True, wai_label='Mean affected'):
    if percent:
        fmt = '{x:.0%}'
    else:
        fmt = '{x:,.2f}'
    # Affectedness indicator
    ax1 = ax
    lines = []
    for tag in tags:
        line = ax1.plot(wai_data.index, wai_data[tag].values, linewidth=lw, label=tag)
        lines.append(line)
    if show_axes:
        ax1.tick_params(axis='y', labelcolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

    # remove border
    if not show_axes:
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)

    # Government stringency index
    ax2 = ax1.twinx()
    line2 = ax2.plot(policy_data['month'].dt.to_timestamp(), policy_data['StringencyIndex_Average'],
                     label='Stringency Index (OxCGRT)', linewidth=lw, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    ax2.set_ylim(0, 100)

    if show_axes:
        ax2.tick_params(axis='y', labelcolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])

    ax2.spines['top'].set_visible(False)
    if not show_axes:
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

    # axvspan workplace closing
    for j, row in policy_data[policy_data['C2M_Workplace closing'] >= 2].iterrows():
        d = row['month'].to_timestamp()
        axvspan1 = ax1.axvspan(d, d + pd.DateOffset(months=1), color='red', alpha=0.15,
                           label='Workplace closing requirement')

    # axvspan stay at home
    for j, row in policy_data[policy_data['C6M_Stay at home requirements'] >= 2].iterrows():
        d = row['month'].to_timestamp()
        axvspan2 = ax1.axvspan(d, d + pd.DateOffset(months=1), color='yellow', alpha=0.15,
                           label='Stay at home requirement')

    if country_counts is not None:
        ax.text(0.77, 0.8, f'n={country_counts[country_counts.country==country].bvdid.iloc[0]:,.0f}',
                transform=ax.transAxes)
    # turn off top ticks
    ax.tick_params(axis='x', top=False, which='both')

    # Set title for each subplot
    if title==True:
        ax.set_title(f'{country}')

    # add legend label for mixed color
    axvspan_color_mix = mix_colors(mpl.colors.to_rgba('red', alpha=0.15), mpl.colors.to_rgba('yellow', alpha=0.15))
    if 'd' in locals():
        axvspan3 = ax1.axvspan(d, d, color=axvspan_color_mix,
                               label='Workplace closing & stay at home requirement')

    # remove grid, axis labels, and axis ticks for all subplots
    if not show_axes:
        for axis in [ax1, ax2]:
            axis.set_yticks([])
        ax.set_ylabel('')
    else:
        if labels:
            ax1.set_ylabel('Share of firms', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax2.set_ylabel('OxCGRT', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        # formatter for ax1
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(fmt))

    try:

        legends = [l[0] for l in lines] + [line2[0], axvspan1, axvspan2, axvspan3]
    except:
        legends = [l[0] for l in lines] + [line2[0]]

    if return_twin_axis:
        return legends, ax2
    else:
        return legends

# test plot country
country = 'United Kingdom'
fig, ax = plt.subplots(figsize=(10, 5))
legends = plot_tags_policy_tracker(ax, country, affected_by_country.reset_index()[affected_by_country.reset_index().country==country].set_index('date')[tags],
                                           oxford_policy_tracker[oxford_policy_tracker.country == coco.convert(country, to='ISO3')], max_affectedness * 1.2,
                                           country_counts=country_counts,
                                           show_axes=True)
fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(1, 0.75))
ax.set_title(country)
plt.tight_layout()
plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//mean_affected_policy_tracker_share_y_corr_2_{country}_llm.pdf', dpi=300,
            bbox_inches='tight')
plt.show()

len(df[df.number_of_employees.notna()])/len(df)

# test plot UK for each industry
agg_dict = {x: 'mean' for x in wai_cols}
agg_dict.update({'cc_data_available': 'sum'})
affected_by_country_industry = df.groupby(['date', 'country', 'nace_section'], observed=True).agg(
    agg_dict).reset_index()
country = 'United Kingdom'
for s in relevant_nace_sections:
    to_plot = affected_by_country_industry[(affected_by_country_industry.nace_section==s) & (affected_by_country_industry.country==country)].set_index('date')
    fig, ax = plt.subplots(figsize=(10, 5))
    legends = plot_tags_policy_tracker(ax, country, to_plot, oxford_policy_tracker[
                                           oxford_policy_tracker.country == coco.convert(country, to='ISO3')],
                                       max_affectedness * 1.2, country_counts=country_counts, show_axes=True)
    fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(1, 0.75))
    ax.set_title(f'{country} - {s}')
    plt.tight_layout()
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//mean_affected_policy_tracker_share_y_corr_2_{country}_{s}_llm.pdf', dpi=300,
                bbox_inches='tight')
    plt.show()


## make a big plot of top countries with affectedness indicator, policy tracker variables
plot_labels = {'affected_3': 'Share of severely affected firms'} # works best
for wai, label in plot_labels.items():
    plot_countries = ['United Kingdom', 'Germany', 'Spain', 'France', 'Netherlands']
    max_affectedness = affected_by_country[wai].loc[:, plot_countries].max()
    fig, axes = plt.subplots(figsize=(7, 10), nrows=5)
    for i,country in enumerate(plot_countries):
        legends = plot_affectedness_policy_tracker(axes[i], country, affected_by_country[wai].loc[:,country],
                                                   oxford_policy_tracker[oxford_policy_tracker.country==coco.convert(country, to='ISO3')],
                                                   max_affectedness=max_affectedness, title=True,
                                                   country_counts=country_counts,
                                                   show_axes=True, labels=True, wai_label=label)
    fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(2, 0.5))
    plt.tight_layout()
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//{wai}_top_5_countries_affectedness_policy_tracker_llm.pdf', dpi=300,
                bbox_inches='tight')
    plt.show()

    plot_states = firm_count_by_state.head(5).index
    max_affectedness = affected_by_state[wai].loc[:, plot_states].max()
    fig, axes = plt.subplots(figsize=(7, 10), nrows=5)
    for i,s in enumerate(plot_states):
        legends = plot_affectedness_policy_tracker(axes[i], s, affected_by_state[wai].loc[:,s],
                                                         oxford_policy_tracker_US[oxford_policy_tracker_US.region_level_1==s],
                                                         max_affectedness=max_affectedness, title=True,
                                                         country_counts=firm_count_by_state.reset_index().rename(columns={'region_level_1': 'country', 'cc_data_available': 'bvdid'}),
                                                         show_axes=True, labels=True, wai_label=label)
    fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(2, 0.5))
    plt.tight_layout()
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//{wai}_top_5_states_affectedness_policy_tracker_llm.pdf', dpi=300,
                bbox_inches='tight')
    plt.show()

## heatmap correlations between tags
correlation_matrix = df[tags].corr()
plt.figure(figsize=(10, 8))
mask = np.eye(correlation_matrix.shape[0], dtype=bool)
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='crest', cbar_kws={'shrink': .8})
plt.title('Heatmap of Pairwise Correlations', size=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# get df of all cities in dataset
# cities = affected_max_by_firm.groupby(['city', 'region_level_1', 'country'], observed=True).size().reset_index(name='firm_count').sort_values('firm_count', ascending=False)
# cities.to_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_orbis_global_cities.parquet')

# country to iso2
country_counts['country_iso2'] = coco.convert(names=country_counts.country, to='ISO2')

# get geocoded cities
cities = pd.read_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_orbis_global_cities_matched_geocoded_fillna.parquet')
cities = cities.merge(country_counts[['country', 'country_iso2']], on='country', how='left')
cities = cities[cities.city.str.len() > 1]
cities = cities[cities.country_iso2.notna()]

# merge with df
cities = cities[['city', 'region_level_1', 'country', 'latitude', 'longitude']].dropna()
cities[['country', 'latitude', 'longitude']] = cities[['country', 'latitude', 'longitude']].astype('category')
df = df.merge(cities[['city', 'region_level_1', 'country', 'latitude', 'longitude']],
                            on=['city', 'region_level_1', 'country'], how='left')
df[['country', 'region_level_1']] = df[['country', 'region_level_1']].astype('category')
memory_usage = df.memory_usage(deep=True)
memory_usage.div(1e9).round(3)
(memory_usage.sum()/1e9).round(3)


# export file for Covid Explorer (Oliver)
# relevant_cols = ['bvdid', 'date', 'country', 'region_level_1', 'city', 'latitude', 'longitude',
#                  'nace_2_digit', 'nace_section', 'number_of_employees', 'content_digest_heartbeat',
#                  'cc_data_available', 'covid_mention', 'affected', 'hygiene_measures', 'remote_work',
#                  'closure']
# float_cols_to_float64(df.loc[df.country.isin(use_countries), relevant_cols]).to_parquet(
#     'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_firm_and_date_geocoded.parquet', index=False)
# del agg_dict['employee_data_available']
# for col in ['demand_affected', 'financial_impact', 'production_affected', 'supply_affected', 'supply_chain_issues', 'travel_restrictions']:
#     del agg_dict[col]
# for col in ['city', 'region_level_1', 'country']:
#     agg_dict[col] = 'first'
# city_date_agg = df[df.country.isin(use_countries)].groupby(['date', 'latitude', 'longitude'], observed=True).agg(agg_dict).reset_index()
# city_date_agg.rename(columns={'cc_data_available': 'firm_count'}, inplace=True)
# city_date_agg = float_cols_to_float64(city_date_agg)
# col_order = ['date', 'city', 'region_level_1',
#        'country', 'latitude', 'longitude', 'firm_count', 'covid_mention', 'affected',
#        'hygiene_measures', 'remote_work', 'closure', 'affected_1',
#        'affected_2', 'affected_3', 'affected_min_1', 'affected_min_2']
# city_date_agg = city_date_agg[col_order]
# city_date_agg.to_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_city_and_date_geocoded_only_relevant_countries_float64.parquet', index=False)
# for ind in relevant_custom_nace_sections:
#     df_ind = df[df.country.isin(use_countries) & (df.nace_custom==ind)].groupby(['date', 'latitude', 'longitude'], observed=True).agg(agg_dict).reset_index()
#     df_ind.rename(columns={'cc_data_available': 'firm_count'}, inplace=True)
#     df_ind = float_cols_to_float64(df_ind[col_order].copy())
#     if ind in sector_name_dict:
#         name = sector_name_dict[ind]
#     else:
#         name = ind
#     df_ind.to_parquet(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_city_and_date_geocoded_only_relevant_countries_float64_{name}.parquet', index=False)

# export files for Oliver (employee weighted)
drop_cols = ['bvdid', 'affected', 'covid_mention', 'production_affected',
       'demand_affected', 'supply_affected', 'hygiene_measures', 'remote_work',
       'supply_chain_issues', 'closure', 'financial_impact',
       'travel_restrictions', 'dataprovider_high_heartbeat', 'content_digest_heartbeat', 'nace_rev_2_core_code_4_digits',
       'nace_2_digit', 'nace_section']
df.drop(columns=drop_cols, inplace=True)
df.loc[~df.cc_data_available, 'number_of_employees'] = np.nan # set number of employees to nan for firm if not analyzed at given date
# exclude_cols = tags + ['demand_affected', 'production_affected', 'supply_affected']
wai_cols = ['affected_1', 'affected_2', 'affected_3'] #[col for col in wai_cols if col not in exclude_cols]
wai_cols_weighted = [col + '_weighted' for col in wai_cols]
df[wai_cols_weighted] = df[wai_cols].multiply(df['number_of_employees'], axis=0).astype('float32')
mem = df.memory_usage(deep=True)
mem.div(1e9).round(3)

agg_dict = {x: 'sum' for x in wai_cols_weighted}
agg_dict.update({'cc_data_available': 'sum', 'number_of_employees': 'sum'})
for col in ['city', 'region_level_1', 'country']:
    agg_dict[col] = 'first'
city_date_agg = df.loc[df.country.isin(use_countries)].groupby(['date', 'latitude', 'longitude'], observed=True).agg(agg_dict).reset_index()
city_date_agg[wai_cols_weighted] = city_date_agg[wai_cols_weighted].div(city_date_agg['number_of_employees'], axis=0)
# replace inf with nan
city_date_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
city_date_agg.rename(columns={'cc_data_available': 'firm_count'}, inplace=True)
city_date_agg[wai_cols_weighted] = city_date_agg[wai_cols_weighted].astype('float64')

# col_order = ['date', 'city', 'region_level_1',
#        'country', 'latitude', 'longitude', 'firm_count', 'covid_mention', 'affected',
#        'hygiene_measures', 'remote_work', 'closure', 'affected_1',
#        'affected_2', 'affected_3', 'affected_min_1', 'affected_min_2']
col_order = ['date', 'city', 'region_level_1', 'country', 'latitude', 'longitude', 'firm_count', 'number_of_employees'] + wai_cols_weighted
city_date_agg = city_date_agg[col_order]
city_date_agg.to_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_city_and_date_geocoded_only_relevant_countries_float64_employee_weighted.parquet', index=False)
city_date_agg = pd.read_parquet('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_city_and_date_geocoded_only_relevant_countries_float64_employee_weighted.parquet')
city_date_agg.affected_1_weighted.isna().sum()/len(city_date_agg)
for ind in relevant_custom_nace_sections:
    df_ind = df[df.nace_custom==ind].groupby(['date', 'latitude', 'longitude'], observed=True).agg(agg_dict).reset_index()
    df_ind[wai_cols_weighted] = df_ind[wai_cols_weighted].div(df_ind['number_of_employees'], axis=0)
    df_ind.rename(columns={'cc_data_available': 'firm_count'}, inplace=True)
    df_ind[wai_cols_weighted] = df_ind[wai_cols_weighted].astype('float64')
    if ind in sector_name_dict:
        name = sector_name_dict[ind]
    else:
        name = ind
    df_ind.to_parquet(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/covid_llm_indicators_by_city_and_date_geocoded_only_relevant_countries_float64_{name}_employee_weighted.parquet', index=False)

# city_date_agg_2 = df[df.cc_data_available].groupby(['date', 'city', 'region_level_1', 'country', 'latitude', 'longitude'], observed=True)[['covid_mention', 'affected']].mean().reset_index()

## country map plots
city_country_aggregates = pd.read_pickle('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/city_country_aggregates_llm.pkl')
# country_counts = pd.read_pickle('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/country_counts.pkl') # with fixed countries
relevant_countries = set(country_counts.head(60).country.to_list()+ ['Ecuador', 'Luxembourg', 'Montenegro', 'Bermuda', 'Liechtenstein'])
relevant_countries = [c for c in relevant_countries if c in use_countries]

## plot for each country
# download GADM dataset to get administrative boundaries for each country
# import requests, zipfile, io, os
# for country in relevant_countries:
#     country_iso3 = coco.convert(names=[country], to='ISO3')
#     if os.path.exists(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/gadm41/{country_iso3}/gadm41_{country_iso3}_1.shp'):
#         continue
#     url = f'https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{country_iso3}_shp.zip'
#     r = requests.get(url)
#     if not zipfile.is_zipfile(io.BytesIO(r.content)):
#         print(f'File for {country} is not a zipfile')
#         continue
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/{country_iso3}')
#     del r, z

country_maps = {}
for i, country in enumerate(relevant_countries):
    country_iso3 = coco.convert(names=[country], to='ISO3')
    map = gpd.read_file(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/gadm41/{country_iso3}/gadm41_{country_iso3}_1.shp')
    country_maps[country] = map

geometry = gpd.points_from_xy(city_country_aggregates.longitude,
                              city_country_aggregates.latitude)
gdf = gpd.GeoDataFrame(city_country_aggregates, geometry=geometry, crs=map.crs)

europe = gpd.read_file(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
europe = europe[europe['SOV_A3'].isin(coco.convert(names=european_countries, to='ISO3'))]
country_maps['Europe'] = europe

def plot_country_map(gdf, country_maps, country, ax, is_us_state=False, alpha=0.7, max_marker_size=5000, colorbar=True, vmax=10, plot_col='affected',
                     colorbar_kwds={'label': 'Mean WAI - Intensity', 'orientation': 'vertical', 'fraction': 0.03, 'pad': 0,
                            'location': 'left', 'shrink': 0.615, 'anchor': (-0.1, 0.5), 'format': '{x:.0%}'}):
    if is_us_state:
        state = country
        map = US_map[US_map.NAME_1 == state].copy()

        # get points in geodf that are in map
        bounds = map.bounds
        minx, miny, maxx, maxy = bounds.minx.min(), bounds.miny.min(), bounds.maxx.max(), bounds.maxy.max()
        to_plot = gdf[(gdf.longitude >= minx) & (gdf.longitude <= maxx) & (gdf.latitude >= miny) & (
                    gdf.latitude <= maxy)].copy()
        to_plot = to_plot[to_plot.region_level_1 == state]
    else:
        to_plot = gdf[gdf.country == country].copy()
        if country=='Europe':
            to_plot = gdf[gdf.country.isin(european_countries)].copy()
            max_marker_size = max_marker_size * 0.1
        map = country_maps[country]

        # merge points that are too close
        to_plot['plot_col_weighted'] = to_plot[plot_col] * to_plot['firm_count']
        to_plot = to_plot.groupby(
                [to_plot.latitude.round(0), to_plot.longitude.round(1)], observed=True,
                as_index=False).agg(
                {'firm_count': 'sum', 'plot_col_weighted': 'sum', 'latitude': 'first',
                 'longitude' : 'first'}).reset_index()
        to_plot = gpd.GeoDataFrame(to_plot, geometry=gpd.points_from_xy(to_plot.longitude, to_plot.latitude), crs=map.crs)
        to_plot[plot_col] = to_plot['plot_col_weighted'] / to_plot['firm_count']

    # make sure that the biggest points are plotted last
    to_plot.sort_values('firm_count', inplace=True)

    firm_count_max = to_plot.firm_count.max()
    marker_sizes_based_on_firm_count = pd.Series(np.power(to_plot['firm_count']/firm_count_max, 0.95))
    try:
        max_marker_size_based_on_firm_count = max(marker_sizes_based_on_firm_count)
    except:
        max_marker_size_based_on_firm_count = 1
    marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count/max_marker_size_based_on_firm_count*max_marker_size

    # simplify map to speed up plotting
    # map = map.simplify(tolerance=0.1)

    # Plotting the world map
    map.plot(color='white', edgecolor='black', ax=ax)

    # Plotting the points on the world map with color based on mean plot_cols
    to_plot.plot(ax=ax, marker='o', alpha=alpha,
             column=plot_col, cmap='YlOrRd',
             markersize=marker_sizes_based_on_firm_count, legend=colorbar, legend_kwds=colorbar_kwds,
             vmax=vmax,
             )

    if country == 'United States of America':
        ax.set_xlim(-129, -60) # only mainland US
        ax.set_ylim(24, 51)
    elif country == 'United Kingdom':
        ax.set_ylim(49, 58) # cut off northern scotland
    elif country == 'Russian Federation':
        ax.set_xlim(19, 95) # cut off eastern russia
        ax.set_ylim(41, 72) # cut off northern russia
    elif country == 'New Zealand':
        ax.set_xlim(163, 179) # cut off at international date line
    elif country == 'Spain':
        ax.set_xlim(-11, 4) # cut off canary islands
        ax.set_ylim(35, 45)
    elif country == 'Europe':
        ax.set_xlim(-27, 40) # cut off at 40 degrees east
        ax.set_ylim(26, 70) # cut off at 70 degrees north
    elif country == 'Australia':
        ax.set_xlim(110, 156)
        ax.set_ylim(-45, -10)

    # relevant_bins = [1, 6, 19, 38]
    # bins_marker_size = pd.cut(marker_sizes_based_on_firm_count, bins=40, precision=0, retbins=True)[1][relevant_bins]
    # bins_firm_count = pd.cut(to_plot['firm_count'], bins=40, precision=0, retbins=True)[1][relevant_bins]
    #
    # # create second legend
    # ax.add_artist(ax.legend(handles=[
    #         mlines.Line2D([], [], color="black", lw=0, marker="o", markersize=np.sqrt(b_marker_size),
    #                 label=['50', '500', '5000', '50000'][i], ) for i, (b_marker_size, b_firm_count) in
    #         enumerate(zip(bins_marker_size, bins_firm_count))],
    #         bbox_to_anchor=(0.07, 0.2), loc="center", title="Firm count", ))

    ax.set_axis_off()

    return ax

# test plot
plot_col = 'affected_3'
fig, axes = plt.subplots(2, 1, figsize=(10, 20))
axes = axes.flatten()
for i, country in enumerate(country_counts.head(2).country):
    colorbar = True if i == len(axes) - 1 else False
    colorbar_kwds = {'label': wai_labels[plot_col], 'orientation': 'horizontal', 'fraction': 0.1, 'pad': 0,
                            'location': 'bottom', 'shrink': 0.9, 'format': '{x:.0%}'}
    plot_country_map(gdf, country_maps, country, axes[i], plot_col=plot_col, colorbar=colorbar, alpha=0.4,
                     colorbar_kwds=colorbar_kwds, vmax=0.2)
    axes[i].set_title(country)
# only show one legend (of last axis)
fig.legend(axes[0].get_legend_handles_labels()[0], axes[0].get_legend_handles_labels()[1], loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.tight_layout()
plt.show()

for plot_col, plot_col_label, vmax in [('affected', 'Mean WAI', 1.5), ('covid_mention', 'Covid mention', 50)]:
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes = axes.flatten()
    for i, country in enumerate(['Germany', 'Netherlands', 'Switzerland']):
        colorbar = True if i == len(axes) - 1 else False
        fmt = '%.1f' if plot_col == 'affected' else '%.0f%%'
        colorbar_kwds = {'label': plot_col_label, 'orientation': 'horizontal', 'fraction': 0.07, 'pad': 0,
                                'location': 'bottom', 'shrink': 0.9, 'format': fmt}
        plot_country_map(gdf, country_maps, country, axes[i], colorbar=colorbar, vmax=vmax, alpha=0.4,
                         colorbar_kwds=colorbar_kwds, plot_col=plot_col)
        axes[i].set_title(country)
    # only show one legend (of last axis)
    fig.legend(axes[0].get_legend_handles_labels()[0], axes[0].get_legend_handles_labels()[1], loc='center right', bbox_to_anchor=(1.4, 0.5))
    plt.tight_layout()
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//affectedness_country_maps_DE_NL_CH_{plot_col}_llm.pdf', dpi=300)
    plt.show()

def plot_time_series_and_map(wai_data, countries, max_affectedness=0.1, map_vmax=0.12, save_path=None):
    fig, axes = plt.subplots(len(countries), 2, figsize=(10, 2 * len(countries)))
    for i, country in enumerate(countries):
        print(country)
        wai_data_country = wai_data[wai_data.country==country].set_index('date')[plot_col]
        if country == 'Europe':
            country_iso3 = 'Europe'
        else:
            country_iso3 = coco.convert(country, to='ISO3')
        legends = plot_affectedness_policy_tracker(axes[i,0], country, wai_data_country, oxford_policy_tracker[
            oxford_policy_tracker.country == country_iso3], max_affectedness=max_affectedness,
                                                   country_counts=country_counts, show_axes=True, labels=False)
        legends = legends[2:]  # remove wai and oxcgrt from legend since axes are labeled
        colorbar_kwds = {'label'   : wai_labels[plot_col], 'orientation': 'horizontal', 'fraction': 0.06,
                         'pad'     : 0, 'location': 'bottom', 'shrink': 0.9, 'format': '{x:.0%}'}
        colorbar = True if i == len(countries) - 1 else False
        plot_country_map(gdf, country_maps, country, axes[i, 1], max_marker_size=300, colorbar=colorbar, plot_col=plot_col,
                         colorbar_kwds=colorbar_kwds, vmax=map_vmax)
        if i == 0:
            fig.legend(legends, [l.get_label() for l in legends], loc='center right',
                       bbox_to_anchor=(0.7, -0.015), ncol=2)
        if i == len(countries) - 1:
            axes[i, 0].set_xticklabels(axes[i, 0].get_xticklabels(), rotation=45, ha='right')
        else:
            axes[i, 0].set_xticklabels([])
        axes[i, 0].tick_params(axis='x', top=False, which='both')
        axes[i, 0].tick_params(axis='x', bottom=False, which='major')
        axes[i, 0].tick_params(axis='x', bottom=True, which='minor')
        axes[i, 0].set_title(f'{country}')
        plt.tight_layout()
        fig.text(0.0, 0.5, wai_labels[plot_col], ha='center', va='center', rotation='vertical',
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        fig.text(0.597, 0.5, 'OxCGRT', ha='center', va='center', rotation='vertical',
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    if save_path:
        plt.savefig(save_path + '.pdf', dpi=300, pad_inches=0.11)
        plt.savefig(save_path + '.png', dpi=300, pad_inches=0.11)
    plt.show()

### Figure 3: Europe + 5 biggest European countries
plot_col = 'affected_3'
top_countries = ['Europe', 'United Kingdom', 'Germany', 'Spain', 'France']
wai_data = affected_by_country.reset_index()
max_affectedness = wai_data[wai_data.country.isin(top_countries)][plot_col].max()
# map_max_affectedness = gdf[gdf.country.isin(top_countries)][plot_col].quantile(0.85)
save_path = 'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots//figure_3_affectedness_policy_tracker_share_y_corr_combined_with_maps_20_02_25'
plot_time_series_and_map(wai_data, top_countries, max_affectedness=max_affectedness, map_vmax=0.12, save_path=save_path)

### plot for each country affectedness vs stringency next to country map (appendix figures - international comparison)
plot_col = 'affected_3'
num_countries = 10
num_batches = 2
top_countries = use_countries
top_countries = [c for c in top_countries if c not in ['Slovenia', 'Hong Kong SAR, China', 'Luxembourg', 'Montenegro', 'Bermuda', 'Liechtenstein']]  #+ ['Ecuador'] # remove slovenia because geocoding does not seem to work well there
wai_data = affected_by_country.reset_index()
max_affectedness = wai_data[wai_data.country.isin(top_countries)][plot_col].max()

for batch_n in range(num_batches):
    start = batch_n * num_countries
    end = (batch_n + 1) * num_countries
    save_path = f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_policy_tracker_share_y_corr_combined_with_maps_{start}_to_{end}_20_02_25'
    plot_time_series_and_map(wai_data, top_countries[start:end], max_affectedness=max_affectedness, map_vmax=0.12,
                             save_path=save_path)

## plot for each US state
firm_count_by_state = df[df.country=='United States of America'].reset_index().groupby('region_level_1', observed=True).bvdid.nunique().sort_values(ascending=False)
states = firm_count_by_state.index.tolist()
firm_count_by_state = firm_count_by_state.reset_index()

# download GADM dataset to get administrative boundaries for each US state
US_map = gpd.read_file(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/gadm41/USA/gadm41_USA_2.shp')
US_map = US_map[US_map.NAME_1.isin(states)]
US_city_country_aggregates = city_country_aggregates[city_country_aggregates.country == 'United States of America'].copy()

geometry_US = gpd.points_from_xy(US_city_country_aggregates.longitude, US_city_country_aggregates.latitude)
gdf_US = gpd.GeoDataFrame(US_city_country_aggregates, geometry=geometry_US, crs=map.crs).dropna(subset=['latitude', 'longitude'])
gdf_US[['longitude', 'latitude']] = gdf_US[['longitude', 'latitude']].astype('float')

# test plot
fig, ax = plt.subplots(1, 1, figsize=(10, 20))
plot_country_map(gdf_US, US_map, 'New York', ax, plot_col='affected_3', vmax=0.3, colorbar=False, is_us_state=True)
plt.show()

## make a big plot of top US states with affectedness indicator, policy tracker variables for US states
firm_count_by_state = firm_count_by_state[firm_count_by_state.region_level_1 != 'District of Columbia'].head(50)

### Figure 4: US + 4 biggest states
plot_states = firm_count_by_state.head(4).region_level_1.tolist()
num_rows = len(plot_states) + 1
plot_col = 'affected_3'
top_countries = ['United States of America']
wai_data = affected_by_country.reset_index()
max_affectedness = max(wai_data[wai_data.country.isin(top_countries)][plot_col].max(),
                       affected_by_state.reset_index()[affected_by_state.reset_index().region_level_1.isin(plot_states)][plot_col].max())
max_affectedness = 0.06 # force y ticks at 2, 4, 6 percent
# map_max_affectedness = gdf[gdf.country.isin(top_countries)][plot_col].quantile(0.85)
save_path = 'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/figure_4_affectedness_policy_tracker_share_y_corr_combined_with_maps_US_states_29_11_24.pdf'

fig, axes = plt.subplots(num_rows, 2, figsize=(10, 2*num_rows))
country = 'United States of America'
wai_data_country = wai_data[wai_data.country == country].set_index('date')[plot_col]
legends = plot_affectedness_policy_tracker(axes[0, 0], country, wai_data_country, oxford_policy_tracker[
    oxford_policy_tracker.country == coco.convert(country, to='ISO3')], max_affectedness=max_affectedness,
                                           country_counts=country_counts, show_axes=True, labels=False)
legends = legends[2:]  # remove wai and oxcgrt from legend since axes are labeled
colorbar_kwds = {'label'   : wai_labels[plot_col], 'orientation': 'horizontal', 'fraction': 0.06, 'pad': 0,
                 'location': 'bottom', 'shrink': 0.9, 'format': '{x:.0%}'}
colorbar = False
plot_country_map(gdf, country_maps, country, axes[0, 1], max_marker_size=300, colorbar=colorbar,
                 plot_col=plot_col, colorbar_kwds=colorbar_kwds, vmax=0.12)
axes[0, 0].set_title(c)
axes[0, 0].tick_params(axis='x', top=False, which='both')
axes[0, 0].tick_params(axis='x', bottom=False, which='major')
axes[0, 0].tick_params(axis='x', bottom=True, which='minor')
axes[0, 0].set_xticklabels([])
for i, state in enumerate(plot_states):
    i += 1 # start from 1
    legends = plot_affectedness_policy_tracker(axes[i, 0], state, affected_by_state[plot_col].rename('affected').loc[:, state], oxford_policy_tracker_US[
        oxford_policy_tracker_US.region_level_1 == state], max_affectedness=max_affectedness, labels=False,
                                               country_counts=firm_count_by_state.rename(
                                                   columns={'region_level_1': 'country'}), show_axes=True)
    legends = legends[2:]  # remove wai and oxcgrt from legend since axes are labeled
    colorbar = True if i == len(axes) - 1 else False
    colorbar_kwds = {'label'   : wai_labels[plot_col], 'orientation': 'horizontal', 'fraction': 0.06,
                     'pad'     : 0, 'location': 'bottom', 'shrink': 0.9, 'format': '{x:.0%}'}
    plot_country_map(gdf_US, US_map, state, axes[i, 1], plot_col='affected_3', max_marker_size=300, colorbar=colorbar,
                     colorbar_kwds=colorbar_kwds, vmax=0.12, is_us_state=True)
    if i == len(axes) - 1:
        axes[i, 0].set_xticklabels(axes[i, 0].get_xticklabels(), rotation=45, ha='right')
    else:
        axes[i, 0].set_xticklabels([])
    axes[i, 0].tick_params(axis='x', top=False, which='both')
    axes[i, 0].tick_params(axis='x', bottom=False, which='major')
    axes[i, 0].tick_params(axis='x', bottom=True, which='minor')
    axes[i, 0].set_title(f'{state}')
fig.legend(legends, [l.get_label() for l in legends], loc='center right',
                   bbox_to_anchor=(0.7, -0.015), ncol=2)
fig.text(0.0, 0.5, wai_labels[plot_col], ha='center', va='center', rotation='vertical',
         color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
fig.text(0.597, 0.5, 'OxCGRT', ha='center', va='center', rotation='vertical',
         color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
plt.savefig(save_path, dpi=300, pad_inches=0.11)
plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, pad_inches=0.11)
plt.tight_layout()
plt.show()

### plot for each state affectedness vs stringency next to country map (appendix figures - US states)
batch_size = 10
num_batches = 5
max_affectedness = 0.06
for batch_n in range(num_batches):
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 2*batch_size))
    start = batch_n * batch_size
    end = (batch_n + 1) * batch_size
    # max_affectedness = share_problem_firms_by_state[share_problem_firms_by_state.region_level_1.isin(
    #     state_counts.region_level_1.iloc[start:end])].any_prob_neg_sent.max()
    for i, state in enumerate(firm_count_by_state.region_level_1.iloc[start:end]):
        legends = plot_affectedness_policy_tracker(axes[i, 0], state,
                                                   affected_by_state[plot_col].rename('affected').loc[:,
                                                   state], oxford_policy_tracker_US[
                                                       oxford_policy_tracker_US.region_level_1 == state],
                                                   max_affectedness=max_affectedness, labels=False,
                                                   country_counts=firm_count_by_state.rename(
                                                           columns={'region_level_1': 'country'}),
                                                   show_axes=True)
        legends = legends[2:]  # remove wai and oxcgrt from legend since axes are labeled
        colorbar = True if i == len(axes) - 1 else False
        colorbar_kwds = {'label': wai_labels[plot_col], 'orientation': 'horizontal', 'fraction': 0.06,
                         'pad'  : 0, 'location': 'bottom', 'shrink': 0.9, 'format': '{x:.0%}'}
        plot_country_map(gdf_US, US_map, state, axes[i, 1], plot_col='affected_3', max_marker_size=300,
                         colorbar=colorbar, colorbar_kwds=colorbar_kwds, vmax=0.12, is_us_state=True)
        if i == len(axes) - 1:
            axes[i, 0].set_xticklabels(axes[i, 0].get_xticklabels(), rotation=45, ha='right')
        else:
            axes[i, 0].set_xticklabels([])
        axes[i, 0].tick_params(axis='x', top=False, which='both')
        axes[i, 0].tick_params(axis='x', bottom=False, which='major')
        axes[i, 0].tick_params(axis='x', bottom=True, which='minor')
        axes[i, 0].set_title(f'{state}')
    fig.legend(legends, [l.get_label() for l in legends], loc='center right', bbox_to_anchor=(0.7, -0.015),
               ncol=2)
    fig.text(0.0, 0.5, wai_labels[plot_col], ha='center', va='center', rotation='vertical',
             color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    fig.text(0.597, 0.5, 'OxCGRT', ha='center', va='center', rotation='vertical',
             color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    plt.tight_layout()
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_policy_tracker_share_y_corr_combined_with_maps_US_states_{start}_to_{end}_29_11.pdf',
                dpi=300, pad_inches=0.11)
    plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_policy_tracker_share_y_corr_combined_with_maps_US_states_{start}_to_{end}_29_11.png',
                dpi=300, pad_inches=0.11)
    plt.show()


## plot on world map
# world = gpd.read_file('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/wb_countries_admin0_10m/wb_countries_admin0_10m.shp')
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
world = world[world.REGION_WB != 'Antarctica']

def plot_world_map_with_cities(ax, alpha=1, plot_col='covid_mention', label='', vmax=10):
    to_plot = city_country_aggregates.copy()
    # merge points that are too close
    to_plot['plot_col_weighted'] = to_plot[plot_col] * to_plot['firm_count']
    to_plot = to_plot[to_plot.country.isin(use_countries)].groupby(
            [to_plot.latitude.round(0), to_plot.longitude.round(1)], observed=True,
            as_index=False).agg(
            {'firm_count': 'sum', 'plot_col_weighted': 'sum', 'latitude': 'first',
             'longitude' : 'first'}).reset_index()

    to_plot[plot_col] = to_plot['plot_col_weighted'] / to_plot['firm_count']
    to_plot.sort_values('firm_count', inplace=True) # make sure that the biggest points are plotted last

    geometry = gpd.points_from_xy(to_plot.longitude, to_plot.latitude)
    to_plot = gpd.GeoDataFrame(to_plot, geometry=geometry, crs=world.crs)

    firm_count_max = to_plot.firm_count.max()
    max_marker_size = 100
    min_marker_size = 0.8
    marker_sizes_based_on_firm_count = pd.Series(np.power(to_plot['firm_count'] / firm_count_max, 0.95))
    marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count * max_marker_size
    # marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count.clip(lower=min_marker_size)

    # to_plot = to_plot[to_plot.firm_count > 1]
    # marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count[to_plot.index]

    world.plot(color='white', edgecolor='black', ax=ax)

    # grey out countries with no data
    world[~world.ISO_A3.isin(coco.convert(names=use_countries, to='iso3'))].plot(color='lightgrey', edgecolor='black', ax=ax, alpha=0.5, label='No data')

    # Plotting the points on the world map with color based on mean any_prob_neg_sent
    to_plot.plot(ax=ax, marker='o',
                 cmap='YlOrRd',
                 column=to_plot[plot_col]*100,
                 # alpha=0.3 + np.sqrt(to_plot['any_prob_neg_sent']).clip(upper=0.7),
                 alpha=alpha,
                 legend=True,
                 vmax=vmax,
                 legend_kwds={'label'   : label, 'orientation': 'vertical', 'fraction': 0.03, 'pad': 0,
                              'location': 'left', 'shrink': 0.615, 'anchor': (-0.1, 0.5),
                                'format': '%.0f%%'},
                 markersize=marker_sizes_based_on_firm_count)

    relevant_bins = [1, 6, 19, 38]
    bins_marker_size = pd.cut(marker_sizes_based_on_firm_count, bins=40, precision=0, retbins=True)[1][relevant_bins]
    bins_firm_count = pd.cut(to_plot['firm_count'], bins=40, precision=0, retbins=True)[1][relevant_bins]

    # create second legend
    ax.add_artist(ax.legend(handles=[
            mlines.Line2D([], [], color="black", lw=0, marker="o", markersize=np.sqrt(b_marker_size),
                    label=['50', '500', '5,000', '50,000'][i], ) for i, (b_marker_size, b_firm_count) in
            enumerate(zip(bins_marker_size, bins_firm_count))],
            bbox_to_anchor=(0.07, 0.2), loc="center", title="Firm count", ))

    # add legend for no data
    from matplotlib import patches as mpatches
    artist = mpatches.Patch(color='lightgrey', label='No data', alpha=0.5)
    ax.add_artist(ax.legend(handles=[artist], bbox_to_anchor=(0.08, 0.055), loc="center"))

    # # add legend for marker size
    # bins_firm_count = [50, 500, 5000, 50000]
    # bins_marker_size = np.power(bins_firm_count/firm_count_max, 0.95)*max_marker_size/10
    # bins_marker_size = pd.cut(marker_sizes_based_on_firm_count, bins=20, retbins=True)[1]
    # bins_marker_size = bins_marker_size[[5, 10, 15, 20]]
    #
    # # create second legend
    # ax.add_artist(ax.legend(handles=[
    #         mlines.Line2D([], [], color="black", lw=0, marker="o", markersize=marker_size,
    #                       label=firm_count) for i, (marker_size, firm_count) in
    #         enumerate(zip(bins_marker_size, bins_firm_count))],
    #         bbox_to_anchor=(0.07, 0.2), loc="center", title="Firm count", ))

    # zoom in to improve visibility
    ax.margins(y=0, x=0)
    ax.set_ylim(ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.set_xlim(ax.get_xlim()[0] + 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ax.get_xlim()[1] - 0.0 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    return ax

plot_col = 'covid_mention'
fig, ax = plt.subplots(figsize=(12, 7))
plot_world_map_with_cities(ax, plot_col=plot_col, label='Covid mention (share of firms)', vmax=40)
plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/{plot_col}_world_map_29_11_24.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/{plot_col}_world_map_29_11_24.png', dpi=300, bbox_inches='tight')
plt.show()

# import plotly.express as px
# fig = px.scatter_geo(city_country_aggregates, lat="latitude", lon="longitude", color="any_prob_neg_sent",
#                      size="firm_count", projection="natural earth")
# # save as png
# fig.write_image('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_04_03_24_plotly.pdf')
# fig.show()
# plt.show()

## plot of world map with affected vs non-affected firm
# country_counts = df.reset_index().groupby('country').bvdid.nunique().sort_values(
#     ascending=False).reset_index()
# country_counts['country_iso2'] = coco.convert(names=country_counts.country.apply(lambda x: x.split('(')[0]), to='ISO2')
# country_name_map = country_counts.set_index('country').country_iso2
# country_name_map = country_name_map[country_name_map != 'not found']
# country_counts = country_counts.groupby('country_iso2').sum().sort_values(ascending=False)
# # country_counts = country_counts.set_index('country_iso2').bvdid.sort_values(ascending=False)
# # max_affectedness_by_firm = df.groupby(level='bvdid')[['covid_mention', 'any_prob_neg_sent']].max()
# # firms_city_country = firms_city_country.join(max_affectedness_by_firm)
# # del max_affectedness_by_firm
#
# ## plot on world map
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# world = world[world.name != 'Antarctica']
#
# affectedness_by_country = df.reset_index().groupby(df.reset_index().country.map(country_name_map))[['covid_mention', 'any_prob_neg_sent']].mean()
# affectedness_by_country.reset_index(inplace=True)
# affectedness_by_country['country_iso3'] = coco.convert(names=affectedness_by_country.country, to='iso3')
# affectedness_by_country = affectedness_by_country.merge(country_counts, left_on='country', right_index=True)
# affectedness_by_country['plot_col'] = affectedness_by_country.any_prob_neg_sent.apply(lambda x: np.log(x + 1))
# affectedness_by_country.loc[affectedness_by_country.bvdid < 2000, 'plot_col'] = np.nan
#
# fig, ax = plt.subplots(figsize=(10, 5))
# plot_df = world.merge(affectedness_by_country,
#                       left_on='iso_a3', right_on='country_iso3')
# plot_df.plot(column='plot_col', ax=ax, legend=False, cmap='OrRd', scheme='quantiles',
#              missing_kwds={'color': 'lightgrey'})  # distinguish between no data and not enough data
# ax.axis('off')
# plt.savefig('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_choropleth_firm_numbers.pdf', dpi=300, bbox_inches='tight')
# plt.show()

## plot choropleth map at 4 different times
# affectedness_by_country = df.reset_index().groupby([df.reset_index().country.map(country_name_map), 'fetch_time'])[['covid_mention', 'any_prob_neg_sent']].mean()
# affectedness_by_country.reset_index(inplace=True)
# affectedness_by_country['country_iso3'] = coco.convert(names=affectedness_by_country.country, to='iso3')
# plot_times = ['2020-02-23', '2020-05-31', '2021-09-22', '2022-12-03']
# affectedness_by_country = affectedness_by_country.merge(country_counts, left_on='country', right_index=True)
# affectedness_by_country['plot_col'] = affectedness_by_country.any_prob_neg_sent.apply(lambda x: np.log(x + 1))
# affectedness_by_country.loc[affectedness_by_country.bvdid < 2000, 'plot_col'] = np.nan
# fig, axes = plt.subplots(2,2, figsize=(10, 5))
# axes = axes.flatten()
# for i, t in enumerate(plot_times):
#     ax = axes[i]
#     plot_df = world.merge(affectedness_by_country[affectedness_by_country.fetch_time == t],
#                           left_on='iso_a3', right_on='country_iso3')
#     plot_df.plot(column='plot_col', ax=ax, legend=False, cmap='OrRd',  # scheme='quantiles',
#                  missing_kwds={'color': 'lightgrey'})  # distinguish between no data and not enough data
#     ax.axis('off')
#     ax.set_title(t)
# plt.savefig('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_choropleth.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# Plotting the world map for 4 different times
# plot_times = ['2020-02-23', '2020-05-31', '2021-09-22', '2022-12-03']
# affectedness_by_country = affectedness_by_country.merge(country_counts, left_on='country', right_index=True)
# affectedness_by_country['plot_col'] = affectedness_by_country.any_prob_neg_sent.apply(lambda x: np.log(x + 1))
# affectedness_by_country.loc[affectedness_by_country.bvdid < 2000, 'plot_col'] = np.nan
# fig, axes = plt.subplots(2,2, figsize=(10, 5))
# axes = axes.flatten()
# for i, t in enumerate(plot_times):
#     ax = axes[i]
#     plot_df = world.merge(affectedness_by_country[affectedness_by_country.fetch_time == t],
#                           left_on='iso_a3', right_on='country_iso3')
#     plot_df.plot(column='plot_col', ax=ax, legend=False, cmap='OrRd',  # scheme='quantiles',
#                  missing_kwds={'color': 'lightgrey'})  # distinguish between no data and not enough data
#     ax.axis('off')
#     ax.set_title(t)
# plt.savefig('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_choropleth.pdf', dpi=300, bbox_inches='tight')
# plt.show()

## scatterplot
country = 'United States of America'
to_plot = problem_firms_by_country.reset_index().pivot(index='country', columns='fetch_time', values='any_prob_neg_sent')
to_plot = to_plot.div(total_firms_by_country.reset_index().set_index('country').bvdid, axis=0)
to_plot = to_plot[to_plot.index.isin(country_counts.head(40).country)]
to_plot.index = [coco.convert(names=[c], to='name_short') for c in to_plot.index]
name_y = 'Share of firms with any problem & negative sentiment, in % (max over all months)'
to_plot = to_plot.max(axis=1).mul(100).to_frame(name=name_y)

ox = oxford_policy_tracker[
    oxford_policy_tracker.country.isin([coco.convert(names=[c], to='ISO3') for c in to_plot.index])]
ox = ox.pivot(index='country', columns='month', values='StringencyIndex_Average')
ox.index = [coco.convert(names=[c], to='name_short') for c in ox.index]
name_x = 'Oxford Policy Tracker Stringency Index (max over all months)'
to_plot[name_x] = ox.max(axis=1)

sns.lmplot(x=name_x, y=name_y, data=to_plot, robust=True, height=7, aspect=1.5)
corr_coefficient = to_plot[name_x].corr(to_plot[name_y])
plt.annotate(f'r: {corr_coefficient:.2f}', xy=(52, 0.1), fontsize=12)
for i, row in to_plot.iterrows():
    plt.annotate(i, (row[name_x], row[name_y]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.tight_layout()
plt.show()

## US states
to_plot = problem_firms_by_US_states.pivot(index='region_level_1', columns='fetch_time', values=label)
to_plot = to_plot.div(total_firms_by_US_states.set_index('region_level_1').total_firms, axis=0)
name_y = 'Share of firms with any problem & negative sentiment, in % (max over all months)'
to_plot = to_plot.max(axis=1).mul(100).to_frame(name=name_y)

ox = oxford_policy_tracker_US.pivot(index='region_level_1', columns='month',
                                    values='StringencyIndex_Average')
name_x = 'Oxford Policy Tracker Stringency Index (max over all months)'
to_plot[name_x] = ox.max(axis=1)

sns.lmplot(x=name_x, y=name_y, data=to_plot, robust=True, height=7, aspect=1.5)
corr_coefficient = to_plot[name_x].corr(to_plot[name_y])
plt.annotate(f'r: {corr_coefficient:.2f}', xy=(62, 2.4), fontsize=12)
for i, row in to_plot.iterrows():
    plt.annotate(i, (row[name_x], row[name_y]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.tight_layout()
plt.show()

## regression table
categories = ['Covid mention', 'Mildly affected', 'Moderately affected', 'Severely affected']

reg_results_path = '/Users/Jakob/Documents/SECO_COVID_Data/output 2024/tables/'
reg_results_sales_path = reg_results_path + 'full_compustat_sample__on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention.csv'
df_coeff = pd.read_csv(reg_results_sales_path, skiprows=lambda x: not x % 2, header=None, nrows=4)
df_std = pd.read_csv(reg_results_sales_path, skiprows=lambda x: (x == 0) or x % 2, header=None, nrows=4).dropna(how='all')


def fix_df_regression_results(df):
    df.index = categories
    df.columns = ['Basic fixed effects', 'Additional controls', 'Full fixed effects']

    # reverse df
    df = df.iloc[::-1]

    return df

df_coeff_sales = fix_df_regression_results(df_coeff[[1,2,3]])
df_std_sales = fix_df_regression_results(df_std[[1,2,3]])
df_coeff_stock_return = fix_df_regression_results(df_coeff[[4,5,6]])
df_std_stock_return = fix_df_regression_results(df_std[[4,5,6]])

# annualize the quarterly coefficients
df_coeff_sales = df_coeff_sales.apply(lambda x: ((1 + x/100) ** 4 - 1) * 100)
df_std_sales = df_std_sales.apply(lambda x: ((1 + x/100) ** 4 - 1) * 100)
df_coeff_stock_return = df_coeff_stock_return.apply(lambda x: ((1 + x/100) ** 4 - 1) * 100)
df_std_stock_return = df_std_stock_return.apply(lambda x: ((1 + x/100) ** 4 - 1) * 100)

categories.reverse()

# Create the plot
plt.rcParams.update({'axes.unicode_minus': False})
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True,
            figsize=(6, 4))

z = 1.96

# Add the horizontal error bars
ax = axes[0]
for i, category in enumerate(categories):
    ax.errorbar(df_coeff_sales.iloc[i]['Basic fixed effects'], i, xerr=z * df_std_sales.iloc[i]['Basic fixed effects'], fmt='o', color='g', label='Basic fixed effects',
                capsize=5)
    ax.errorbar(df_coeff_sales.iloc[i]['Additional controls'], i-0.2, xerr=z * df_std_sales.iloc[i]['Additional controls'], fmt='o', color='b',label='Additional controls',
                capsize=5)
    ax.errorbar(df_coeff_sales.iloc[i]['Full fixed effects'], i-0.4, xerr=z * df_std_sales.iloc[i]['Full fixed effects'], fmt='o', color='r', label='Full fixed effects',
                capsize=5)

ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)
ax.set_xlim(-25, 9)
ax.set_xlabel('Annualized sales growth')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

ax = axes[1]
for i, category in enumerate(categories):
    ax.errorbar(df_coeff_stock_return.iloc[i]['Basic fixed effects'], i, xerr=z * df_std_stock_return.iloc[i]['Basic fixed effects'], fmt='o', color='g', label='Basic fixed effects',
                capsize=5)
    ax.errorbar(df_coeff_stock_return.iloc[i]['Additional controls'], i-0.2, xerr=z * df_std_stock_return.iloc[i]['Additional controls'], fmt='o', color='b', label='Additional controls',
                capsize=5)
    ax.errorbar(df_coeff_stock_return.iloc[i]['Full fixed effects'], i-0.4, xerr=z * df_std_stock_return.iloc[i]['Full fixed effects'], fmt='o', color='r', label='Full fixed effects',
                capsize=5)

ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)
ax.set_xlim(-8, 0.5)
ax.set_xlabel('Annualized stock return')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

for ax in axes:
    ax.grid(True, which='both', axis='x', linewidth=0.15)
    # # minor grid with half the linewidth
    # ax.grid(True, which='minor', axis='x', linewidth=0.25)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.tick_params(axis='y', which='minor', left=False)
    ax.tick_params(axis='y', which='minor', right=False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[-3:], labels[-3:], loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=True, #0.06
              shadow=True, ncol=3, bbox_transform=plt.gcf().transFigure, columnspacing=0.5,
           handletextpad=-0.2, title='95% confidence intervals')
plt.tight_layout()

# Show the plot
plt.subplots_adjust(wspace=0.18)
fig.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/regression_errorbars_17-10-24.pdf', dpi=300, bbox_inches='tight')
plt.show()

# high heartbeat filter
# high_heartbeat_bvdids = df[df.content_digest_heartbeat==1].bvdid.unique()

## make side panel plot
# dtypes = affected_max_by_firm.dtypes
# sum_cols = dtypes[dtypes!='category'].index
sum_cols = ['covid_mention', 'cc_data_available'] + affected_dummy_cols + tags
affected_by_country_max = affected_max_by_firm.groupby('country', observed=True)[sum_cols].sum()
affected_by_country_max['affected_min_1'] = affected_by_country_max[affected_dummy_cols].sum(axis=1, min_count=1)
affected_by_country_max['affected_min_2'] = affected_by_country_max[affected_dummy_cols[1:]].sum(axis=1, min_count=1)

def plot_affectedness_categories_by_country_side_panel(ax, n_countries=17, drop_countries=['Japan', 'China'],
                                                       order_by_n_firms=False):
    countries_to_plot = country_counts[country_counts.country.isin(use_countries)].head(n_countries).country
    countries_to_plot = [c for c in countries_to_plot if c not in drop_countries]
    to_plot = affected_by_country_max.loc[countries_to_plot].copy(deep=True)
    if order_by_n_firms:
        to_plot['sort_col'] = to_plot.cc_data_available
    else:
        to_plot['sort_col'] = to_plot.affected_3/to_plot.cc_data_available
    to_plot.sort_values('sort_col', ascending=True, inplace=True)

    affectedness_cols = ['covid_mention'] + affected_dummy_cols[::-1]

    labels = {'covid_mention': 'Covid mention', 'affected_3': 'Severely affected',
              'affected_2':'Moderately affected', 'affected_1': 'Slightly affected'}

    offset_x = pd.Series(0, index=to_plot.index)
    for i, col in enumerate(affectedness_cols): #  + ['bvdid']
        to_plot[col] = to_plot[col].div(to_plot['cc_data_available'])*100 # calculate shares
        if col == 'covid_mention':
            color = 'black'
            offset_y = -0.2
            ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], color=color, height=0.4,
                    label=labels[col])
        else:
            color = sns.color_palette('YlOrRd_r_d')[1 + (i-1)*2]
            offset_y = 0.2
            ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], left=offset_x, height=0.4,
                    color=color,  label=labels[col])
            offset_x += to_plot[col]

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.9, 1), bbox_transform=ax.transAxes,
                       ncol=2, fancybox=True, shadow=True) #, title='WAIs')

    ax.set_yticks(range(len(to_plot)))
    ax.set_yticklabels(to_plot.index, ha='left')
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', which='both', top=False)
    ax.margins(y=0.02)
    ax.set_xlabel('Share of firms')

    # remove y ticks
    ax.tick_params(axis='y', which='both', left=False, right=True)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax.grid(True, which='major', axis='x', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', linewidth=0.15)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 50)

    return ax


fig, ax = plt.subplots(figsize=(4.3, 5.5))
plot_affectedness_categories_by_country_side_panel(ax)
plt.tight_layout()
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_small_llm.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_small_llm.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 13))
plot_affectedness_categories_by_country_side_panel(ax, n_countries=50)
plt.tight_layout()
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_large_llm.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_large_llm.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 13))
plot_affectedness_categories_by_country_side_panel(ax, n_countries=50, order_by_n_firms=True)
plt.tight_layout()
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_large_llm_sorted_n_firms.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_country_large_llm_sorted_n_firms.png', dpi=300, bbox_inches='tight')
plt.show()

def plot_affectedness_tags_by_country_with_affected(ax, n_countries=17, drop_countries=['Japan', 'China'],
                                      order_by_n_firms=True, use_tags=['hygiene_measures', 'remote_work',
                                      'supply_chain_issues', 'closure', 'financial_impact', 'travel_restrictions']):
    countries_to_plot = country_counts[country_counts.country.isin(use_countries)].head(n_countries).country
    countries_to_plot = [c for c in countries_to_plot if c not in drop_countries]
    to_plot = affected_by_country_max.loc[countries_to_plot].copy(deep=True)
    if order_by_n_firms:
        to_plot['sort_col'] = to_plot.bvdid
    else:
        to_plot['sort_col'] = to_plot.affected_min_1/to_plot.bvdid
    to_plot.sort_values('sort_col', ascending=True, inplace=True)

    plot_cols = ['affected_min_1'] + use_tags

    labels = {t: ' '.join(w.capitalize() for w in t.split('_')) for t in use_tags}
    labels.update({'affected_min_1': 'At least mildly affected'})

    offset_y = 0
    for i, col in enumerate(plot_cols):
        to_plot[col] = to_plot[col].div(to_plot['bvdid'])*100 # calculate shares
        if col=='affected_min_1':
            color = 'black'
        else:
            color = sns.color_palette()[i]
        offset_y += 0.15
        ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], height=0.15,
                color=color, label=labels[col])

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.9, 1), bbox_transform=ax.transAxes,
                       ncol=2, fancybox=True, shadow=True) #, title='WAIs')

    ax.set_yticks(range(len(to_plot)))
    ax.set_yticklabels(to_plot.index, ha='left')
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', which='both', top=False)
    ax.margins(y=0.02)
    ax.set_xlabel('Share of firms')

    # remove y ticks
    ax.tick_params(axis='y', which='both', left=False, right=True)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax.grid(True, which='major', axis='x', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', linewidth=0.15)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 50)

    return ax

def plot_affectedness_tags_by_group(ax, to_plot, order_by_n_firms=True,
                                    use_tags=['hygiene_measures', 'remote_work', 'supply_chain_issues',
                                              'closure', 'financial_impact', 'travel_restrictions']):
    if order_by_n_firms:
        to_plot['sort_col'] = to_plot.cc_data_available
    else:
        to_plot['sort_col'] = to_plot.affected_min_1 / to_plot.cc_data_available
    to_plot.sort_values('sort_col', ascending=True, inplace=True)

    plot_cols = use_tags

    labels = {t: ' '.join(w.capitalize() for w in t.split('_')) for t in use_tags}

    offset_y = -0.15*(len(plot_cols)//2)
    for i, col in enumerate(plot_cols):
        to_plot[col] = to_plot[col].div(to_plot['affected_min_1'])*100 # calculate shares
        color = sns.color_palette()[i]
        offset_y += 0.15
        ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], height=0.15,
                color=color, label=labels[col])

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.9, 1), bbox_transform=ax.transAxes,
                       ncol=2, fancybox=True, shadow=True) #, title='WAIs')

    ax.set_yticks(range(len(to_plot)))
    ax.set_yticklabels(to_plot.index, ha='left')
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', which='both', top=False)
    ax.margins(y=0.02)
    ax.set_xlabel('Share among affected firms')

    # remove y ticks
    ax.tick_params(axis='y', which='both', left=False, right=True)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax.grid(True, which='major', axis='x', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', linewidth=0.15)
    ax.set_axisbelow(True)

    return ax

## tags by country plot
n_countries = 13
drop_countries = ['Japan', 'China']
countries_to_plot = country_counts[country_counts.country.isin(use_countries)].head(n_countries).country
countries_to_plot = [c for c in countries_to_plot if c not in drop_countries]
to_plot = affected_by_country_max.loc[countries_to_plot].copy(deep=True)
fig, ax = plt.subplots(figsize=(4.3, 5.5))
plot_affectedness_tags_by_group(ax, to_plot)
plt.tight_layout()
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_tags_by_country_small_llm_20_2_25.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_tags_by_country_small_llm_20_2_25.png', dpi=300, bbox_inches='tight')
plt.show()

## max affectedness by industry
affected_by_industry_max = affected_max_by_firm.groupby('nace_section', observed=True)[sum_cols].sum()
affected_by_industry_max['affected_min_1'] = affected_by_industry_max[affected_dummy_cols].sum(axis=1, min_count=1)
affected_by_industry_max['affected_min_2'] = affected_by_industry_max[affected_dummy_cols[1:]].sum(axis=1, min_count=1)

## tags by industry plot
to_plot = affected_by_industry_max.loc[relevant_nace_sections].copy(deep=True)
fig, ax = plt.subplots(figsize=(4.3, 5.5))
plot_affectedness_tags_by_group(ax, to_plot)
plt.tight_layout()
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_tags_by_industry_small_llm.pdf', dpi=300, bbox_inches='tight')
plt.show()

## max affectedness by industry weighted by number of employees
to_plot = affected_by_industry_max_weighted.loc[relevant_custom_nace_sections].copy(deep=True)
fig, ax = plt.subplots(figsize=(4.3, 5.5))
plot_affectedness_tags_by_group(ax, to_plot)
plt.tight_layout()
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_tags_by_industry_small_llm_weighted.pdf', dpi=300, bbox_inches='tight')
plt.show()

## max affectedness by industry weighted by number of employees for only severely affected firms
affected_by_industry_max_weighted_severe_only = affected_max_by_firm.copy()
affected_by_industry_max_weighted_severe_only.loc[affected_by_industry_max_weighted_severe_only.affected_3==False, tags] = 0
affected_by_industry_max_weighted_severe_only[wai_cols] = affected_by_industry_max_weighted_severe_only[wai_cols].multiply(affected_by_industry_max_weighted_severe_only['number_of_employees'], axis=0)
affected_by_industry_max_weighted_severe_only['employee_data_available'] = affected_by_industry_max_weighted_severe_only['number_of_employees'].notna()
agg_dict = {x: 'sum' for x in wai_cols}
agg_dict.update({'cc_data_available': 'sum', 'number_of_employees': 'sum', 'employee_data_available': 'sum'})
affected_by_industry_max_weighted_severe_only = affected_by_industry_max_weighted_severe_only.groupby('nace_custom', observed=True).agg(agg_dict)
affected_by_industry_max_weighted_severe_only[wai_cols] = affected_by_industry_max_weighted_severe_only[wai_cols].div(affected_by_industry_max_weighted_severe_only['number_of_employees'], axis=0)

to_plot = affected_by_industry_max_weighted_severe_only.loc[relevant_custom_nace_sections].copy(deep=True)
fig, ax = plt.subplots(figsize=(4.3, 5.5))
plot_affectedness_tags_by_group(ax, to_plot)
plt.tight_layout()
plt.savefig('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_tags_by_industry_small_llm_weighted_severe_only.pdf', dpi=300, bbox_inches='tight')
plt.show()

## make US state plot with different affectedness indicators
# affectedness_categories_by_us_state = df[df.country=='United States of America'].reset_index().groupby(['region_level_1', 'date'], observed=True)[affectedness_cols].sum()
# affectedness_categories_by_us_state = affectedness_categories_by_us_state.groupby('region_level_1').max()
# affectedness_categories_by_us_state = affectedness_categories_by_us_state.join(firm_count_by_state.set_index('region_level_1'))
# affectedness_categories_by_us_state.sort_values('bvdid', ascending=True, inplace=True)

affected_by_us_state_max = affected_max_by_firm[affected_max_by_firm.country=='United States of America'].groupby('region_level_1', observed=True)[sum_cols].sum()

def plot_affectedness_categories_by_us_state_side_panel_shares_appendix(ax, n_states=50, drop_states=[],
                                                               order_by_n_firms=False):
    to_plot = affected_by_us_state_max.dropna(subset=['cc_data_available']).tail(n_states).copy(deep=True)
    to_plot.drop(drop_states, inplace=True)

    if order_by_n_firms:
        to_plot['sort_col'] = to_plot.cc_data_available
    else:
        to_plot['sort_col'] = to_plot.affected_3 / to_plot.cc_data_available
    to_plot.sort_values('sort_col', ascending=True, inplace=True)

    affectedness_cols = ['covid_mention'] + affected_dummy_cols[::-1]

    labels = {'covid_mention': 'Covid mention', 'affected_3': 'Severely affected',
              'affected_2'   : 'Moderately affected', 'affected_1': 'Slightly affected'}

    offset_x = pd.Series(0, index=to_plot.index)
    for i, col in enumerate(affectedness_cols):  # + ['cc_data_available']
        to_plot[col] = to_plot[col].div(to_plot['cc_data_available']) * 100  # calculate shares
        if col == 'covid_mention':
            color = 'black'
            offset_y = -0.2
            ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], color=color, height=0.4,
                    label=labels[col])
        else:
            color = sns.color_palette('YlOrRd_r_d')[1 + (i - 1) * 2]
            offset_y = 0.2
            ax.barh(y=np.arange(len(to_plot)) - offset_y, width=to_plot[col], left=offset_x, height=0.4,
                    color=color, label=labels[col])
            offset_x += to_plot[col]

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), bbox_transform=ax.transAxes, ncol=2,
                       fancybox=True, shadow=True)  # , title='WAIs')

    ax.set_yticks(range(len(to_plot)))
    ax.set_yticklabels(to_plot.index, ha='left')
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', which='both', top=False)
    ax.margins(y=0.02)
    ax.set_xlabel('Share of firms')

    # remove y ticks
    ax.tick_params(axis='y', which='both', left=False, right=True)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax.grid(True, which='major', axis='x', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', linewidth=0.15)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 50)

    return ax

## make side panel for all states for appendix
fig, ax = plt.subplots(figsize=(6, 13))
plot_affectedness_categories_by_us_state_side_panel_shares_appendix(ax)
plt.tight_layout()
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_us_state_appendix_llm.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/sidepanel_affectedness_categories_by_us_state_appendix_llm.png', dpi=300, bbox_inches='tight')
plt.show()

## make world map plot with side panel
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[5, 1], height_ratios=[1, 1])
plot_world_map_with_cities(ax=axes[0])
plot_affectedness_categories_by_country_side_panel(ax=axes[1])
plt.tight_layout()
plt.savefig('C:/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_12_02_24_with_side_panel.pdf', dpi=300, bbox_inches='tight')
plt.show()

## world map with covid mention
city_country_aggregates['covid_mention_weighted'] = city_country_aggregates['covid_mention'] * \
                                                        city_country_aggregates['firm_count']
def plot_world_map_with_cities_covid_mention(ax, alpha=1):
    # merge points that are too close
    to_plot = city_country_aggregates[city_country_aggregates.country.isin(use_countries)].groupby(
            [city_country_aggregates.latitude.round(0), city_country_aggregates.longitude.round(1)],
            as_index=False).agg(
            {'firm_count': 'sum', 'covid_mention_weighted': 'sum', 'latitude': 'first',
             'longitude' : 'first'}).reset_index()

    to_plot['covid_mention'] = to_plot['covid_mention_weighted'] / to_plot['firm_count']
    to_plot.sort_values('firm_count', inplace=True) # make sure that the biggest points are plotted last

    geometry = gpd.points_from_xy(to_plot.longitude, to_plot.latitude)
    to_plot = gpd.GeoDataFrame(to_plot, geometry=geometry, crs=world.crs)

    firm_count_max = to_plot.firm_count.max()
    max_marker_size = 100
    min_marker_size = 0.8
    marker_sizes_based_on_firm_count = pd.Series(np.power(to_plot['firm_count'] / firm_count_max, 0.95))
    marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count * max_marker_size
    # marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count.clip(lower=min_marker_size)

    # to_plot = to_plot[to_plot.firm_count > 1]
    # marker_sizes_based_on_firm_count = marker_sizes_based_on_firm_count[to_plot.index]

    world.plot(color='white', edgecolor='black', ax=ax)

    # grey out countries with no data
    no_data = world[~world.name.apply(lambda x: coco.convert(x, to='iso3')).isin(coco.convert(names=use_countries, to='iso3'))].plot(color='lightgrey', edgecolor='black', ax=ax, alpha=0.5, label='No data')

    # Plotting the points on the world map with color based on mean any_prob_neg_sent
    to_plot.plot(ax=ax, marker='o',
                 cmap='YlOrRd',
                 column=to_plot['covid_mention']*100,
                 # alpha=0.3 + np.sqrt(to_plot['any_prob_neg_sent']).clip(upper=0.7),
                 alpha=alpha,
                 legend=True,
                 vmax=40,
                 legend_kwds={'label'   : 'Covid mention (share of firms)', 'orientation': 'vertical', 'fraction': 0.03, 'pad': 0,
                              'location': 'left', 'shrink': 0.615, 'anchor': (-0.1, 0.5),
                                'format': '%.0f%%'},
                 markersize=marker_sizes_based_on_firm_count)

    relevant_bins = [1, 6, 19, 38]
    bins_marker_size = pd.cut(marker_sizes_based_on_firm_count, bins=40, precision=0, retbins=True)[1][relevant_bins]
    bins_firm_count = pd.cut(to_plot['firm_count'], bins=40, precision=0, retbins=True)[1][relevant_bins]

    # create second legend
    ax.add_artist(ax.legend(handles=[
            mlines.Line2D([], [], color="black", lw=0, marker="o", markersize=np.sqrt(b_marker_size),
                    label=['50', '500', '5,000', '50,000'][i], ) for i, (b_marker_size, b_firm_count) in
            enumerate(zip(bins_marker_size, bins_firm_count))],
            bbox_to_anchor=(0.07, 0.2), loc="center", title="Firm count", ))

    # add legend for no data
    from matplotlib import patches as mpatches
    artist = mpatches.Patch(color='lightgrey', label='No data', alpha=0.5)
    ax.add_artist(ax.legend(handles=[artist], bbox_to_anchor=(0.08, 0.055), loc="center"))

    # zoom in to improve visibility
    ax.margins(y=0, x=0)
    ax.set_ylim(ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.set_xlim(ax.get_xlim()[0] + 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ax.get_xlim()[1] - 0.0 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

    return ax



fig, ax = plt.subplots(figsize=(12, 7))
plot_world_map_with_cities_covid_mention(ax)
plt.savefig('C:/UsersJakob/Documents/SECO_COVID_Data/output 2024/plots/affectedness_world_map_18_04_24_covid_mention.pdf', dpi=300, bbox_inches='tight')
plt.show()

## create table with share of firms covered
msme = pd.read_excel('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/2019 MSME-EI Database.xlsx',
                     sheet_name='Latest Year Available', header=[0,1])
msme.dropna(subset=[('Country Code', 'Unnamed: 3_level_1')], inplace=True)
num_firms_cols = [('Number of Enterprises', c) for c in ['Micro', 'Small', 'Medium', 'Large']]
msme['total_firms'] = msme[num_firms_cols].sum(axis=1, skipna=False)
msme[('MSME Definitions \n(number of employees)', 'Micro')].value_counts().head(17)
right_defs = ['0-5', '<6', '1-5', '≤5', '0-4', '<5', '1-4', '0-3', '1-3', '1-4, 05_82_LESS_642']
msme['def_order'] = msme[('MSME Definitions \n(number of employees)', 'Micro')].apply(
    lambda x: right_defs.index(x) if x in right_defs else np.nan)
msme['total_firms_is_nan'] = msme['total_firms'].isna()
msme.sort_values(['total_firms_is_nan', 'def_order', ('Year', 'Unnamed: 2_level_1')],
                 ascending=[True, True, False], inplace=True)
msme.drop_duplicates(subset=('Country Code', 'Unnamed: 3_level_1'), keep='first', inplace=True)
msme.set_index(('Country Code', 'Unnamed: 3_level_1'), inplace=True)
msme.sort_index(inplace=True)
msme['micro_definition'] = msme[[('MSME Definitions \n(number of employees)', 'Micro')]]
num_firms_cols_rename_dict = {col: 'num_firms_' + col[1].lower() for col in num_firms_cols}
for old, new in num_firms_cols_rename_dict.items():
    msme[new] = msme[old]
msme.drop(columns=num_firms_cols, inplace=True)
msme.columns = msme.columns.droplevel(1)
keep_cols = ['Year', 'Population, total', 'Source of MSME Data', 'Website', 'total_firms', 'micro_definition'] + \
            list(num_firms_cols_rename_dict.values())
msme = msme[keep_cols].copy()
msme['num_firms_except_micro'] = msme[list(num_firms_cols_rename_dict.values())].drop(columns='num_firms_micro').sum(axis=1, skipna=False)
msme.sort_values('num_firms_except_micro', ascending=False, inplace=True)

country_firms_table = country_counts[country_counts.bvdid>=100].copy()
country_firms_table['country_iso3'] = country_firms_table['country'].astype(str).apply(lambda x: coco.convert(names=[x], to='ISO3'))
country_firms_table.rename(columns={'bvdid': 'num_firms_analyzed'}, inplace=True)
country_firms_table = country_firms_table[country_firms_table.country_iso3.apply(type)!=list].copy()
country_firms_table.drop_duplicates(subset='country_iso3', keep='first', inplace=True)
country_firms_table.set_index('country_iso3', inplace=True)
country_firms_table = country_firms_table.join(msme)

sbds = pd.read_csv('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/SDBS_BDI_ISIC4_05042024095950225.csv', sep=',')
sbds = sbds[sbds.Time<=2020].copy()
sbds.dropna(subset=['Value'], inplace=True)
sbds = sbds[sbds.SEC=='05_82_LESS_642'].copy()
sbds.sort_values(['LOCATION', 'Time'], ascending=[True, False], inplace=True)
sbds.drop_duplicates(subset=['LOCATION', 'IND', 'SCL'], keep='first', inplace=True)
sbds = sbds.pivot(index='LOCATION', columns='SCL', values='Value')
sbds['num_firms_except_micro'] = sbds['TOTAL'] - sbds['1-4']
sbds = sbds[['num_firms_except_micro']].copy()
sbds.dropna(inplace=True)
sbds['Source of MSME Data'] = 'OECD'
sbds['Year'] = 2020
sbds['Website'] = 'https://stats.oecd.org/Index.aspx?DataSetCode=SDBS_BDI_ISIC4'
sbds['micro_definition'] = '1-4, 05_82_LESS_642'
country_firms_table.update(sbds)

manual_data = {'index': ['USA', 'GBR', 'KOR'],
               'num_firms_except_micro': [2312130, 557370, 853449],
               'micro_definition': ['0-4', '0-4', '0-4'],
               'Year': [2020, 2020, 2019],
               'Source of MSME Data': ['U.S. Census Bureau', 'UK Statistics Authority', 'Statistics Korea'],
               'Website': ['https://www2.census.gov/programs-surveys/susb/tables/2020/us_state_naics_detailedsizes_2020.xlsx',
                           'https://assets.publishing.service.gov.uk/media/5f7439c48fa8f5188dad0e85/BPE__2020_detailed_tables.xlsx',
                           'https://kostat.go.kr/board.es?mid=a20104040000&bid=11726&act=view&list_no=388277',
                            ]
               }
manual_data = pd.DataFrame(manual_data).set_index('index')
country_firms_table.update(manual_data)

country_firms_table['share_firms_analyzed_total'] = country_firms_table['num_firms_analyzed'] / country_firms_table['total_firms']
country_firms_table['share_firms_analyzed_total_except_micro'] = country_firms_table['num_firms_analyzed'] / country_firms_table['num_firms_except_micro']
country_firms_table['num_firms_analyzed_per_m_pop'] = country_firms_table['num_firms_analyzed'] / (country_firms_table['Population, total']/1e6)
country_firms_table['right_micro_def'] = country_firms_table['micro_definition'].apply(lambda x: x in right_defs)
country_firms_table['share_firms_analyzed_total_except_micro_table'] = country_firms_table['share_firms_analyzed_total_except_micro'].apply(lambda x: f'{x:.2f}' if not np.isnan(x) else '')
country_firms_table['num_firms_except_micro'] = country_firms_table['num_firms_except_micro'].apply(lambda x: f'{x:,.0f}' if not np.isnan(x) else '')
country_firms_table.loc[~country_firms_table['right_micro_def'] & country_firms_table['share_firms_analyzed_total_except_micro'].notna(), 'num_firms_except_micro'] += '*'
country_firms_table.loc[country_firms_table['micro_definition'].astype(str).str.contains('LESS') & country_firms_table['share_firms_analyzed_total_except_micro'].notna(), 'num_firms_except_micro'] += '$\\dagger$'
country_firms_table['Source of MSME Data'] = country_firms_table['Source of MSME Data'].str.split(' - ').str[0]
country_firms_table['Source of MSME Data'] = country_firms_table['Source of MSME Data'].str.split(',').str[0]
country_firms_table['Source of MSME Data'] = country_firms_table['Source of MSME Data'].str.split('(').str[0]
country_firms_table['Source of MSME Data'].str.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s').str[0]
country_firms_table['Source of MSME Data'] = country_firms_table['Source of MSME Data'].str.split('. Annual ').str[0].str.strip()
country_firms_table['source'] = country_firms_table['Source of MSME Data'] + ' (\href{' + country_firms_table.Website + '}{' + country_firms_table.Year.apply(lambda x: str(int(x)) if not np.isnan(x) else '') + '})'
country_firms_table.loc[country_firms_table['num_firms_except_micro'].isna(), 'source'] = ''

# merge with oecd share of people using internet
country_firms_table = country_firms_table.join(oecd[['country_iso3', 'share_people_using_internet']].set_index('country_iso3'))

rename_dict = {'country': 'Country', 'num_firms_analyzed': 'Firms analyzed',
                'num_firms_analyzed_per_m_pop': 'Firms analyzed per million inhabitants',
               # 'total_firms': 'Total number of firms',
               # 'share_firms_analyzed_total': 'Share of total firms analyzed',
                'num_firms_except_micro': 'Total firms (excluding $<$5 employees)',
               'share_firms_analyzed_total_except_micro_table': 'Share of firms analyzed',
               # 'share_firms_analyzed_total_except_micro': 'Share of firms analyzed',
               # 'micro_definition': 'Definition of micro firms (num. employees)',
               # 'Population, total': 'Population',
               # 'Year': 'Year of firm demographics data',
               'source': 'Source of firm demographics data',
               'share_people_using_internet': 'Share of population using internet',
               }

import re
export_table = country_firms_table.head(69)[list(rename_dict.keys())].rename(columns=rename_dict).set_index('Country')
export_table['Share of firms analyzed'] = pd.to_numeric(export_table['Share of firms analyzed'])*100

# split in two parts, cut at 42
for i, table in enumerate([export_table.head(42), export_table.tail(27)]):
    s = table.style.format({"Firms analyzed"                            : "{:,.0f}",
                                   "Total number of firms"                     : "{:,.0f}",
                                   "Share of total firms analyzed"             : "{:.2f}",
                                   # "Total firms (excluding $<$5 employees)": "{:,.0f}",
                                   "Share of firms analyzed": "{:.0f}",
                                   "Population"                                : "{:,.0f}",
                                   "Firms analyzed per million inhabitants"    : "{:,.0f}",
                                   "Year of firm demographics data"            : "{:.0f}",
                                   "Share of population using internet"        : "{:.0f}",
                                   "Definition of micro firms (num. employees)": lambda x: str(x)[:4]},
            na_rep='')

    s.highlight_between(subset='Firms analyzed per million inhabitants', right=1000, props='color:red')
    # s.background_gradient(cmap='PuBu', axis=0, subset='Share of firms analyzed', vmin=0, vmax=1)
    s.highlight_between(subset='Share of firms analyzed', right=20, props='color:red')
    s.highlight_between(subset='Share of population using internet', right=82.7, props='color:red')

    path = (f'/Users/Jakob/Documents/SECO_COVID_Data/output 2024/tables/country_firms_table_llm_21_2_25_{i}.tex')
    s.to_latex(path, hrules=True, convert_css=True,
               # environment='longtable', # clines='all;data', environment='longtable'
               multicol_align="c",
               column_format='p{2.8cm}|p{1.3cm}|p{1.47cm}|p{1.4cm}|p{1.2cm}|p{3.7cm}|p{1.46cm}',
               # + 'p{3cm}|'.join(['']*(len(rename_dict))))
               )

    # replace each occurence of '\color{red} NUMBER' with '\textcolor{red}{NUMBER}' to fix vertical alignment
    with open(path, 'r') as file:
        filedata = file.read()

        pattern = r'\\color\{(red)\} (\d+) '
        replacement = r'\\textcolor{\1}{\2} '

        filedata = re.sub(pattern, replacement, filedata)

    with open(path, 'w') as file:
        file.write(filedata)



country_firms_table.to_csv('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/tables/country_firms_table.csv')
country_firms_table = pd.read_csv('/Users/Jakob/Documents/SECO_COVID_Data/output 2024/tables/country_firms_table.csv')

# oecd share of businesses with website
# oecd = 'C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/oecd-share-of-businesses-with-website.xlsx'
# oecd = pd.read_excel(oecd, sheet_name='Table', skiprows=5, skipfooter=2, usecols='B:W')
# oecd.drop(columns=['Time period.1'], inplace=True)
# oecd.rename(columns={'Time period': 'country'}, inplace=True)
# oecd.country = oecd.country.str.replace('·\u2007\u2007', '').str.strip()
# oecd.set_index('country', inplace=True)
# oecd.dropna(how='all', inplace=True)
# oecd.columns = ['year_' + str(x) for x in oecd.columns]
# oecd = pd.wide_to_long(oecd.reset_index(), stubnames='year_', i='country', j='year').sort_index().dropna().reset_index()
# oecd.drop_duplicates(subset=['country'], inplace=True, keep='last')
# oecd.rename(columns={'year_': 'share_businesses_with_website'}, inplace=True)
# oecd.to_csv('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/oecd-share-of-businesses-with-website.csv', index=False)
# oecd = pd.read_csv('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/oecd-share-of-businesses-with-website.csv')
#
# oecd['country_iso3'] = oecd['country'].astype(str).apply(lambda x: coco.convert(names=[x], to='ISO3'))
# country_firms_table = country_firms_table.set_index('country_iso3').join(oecd.set_index('country_iso3').drop(columns='country'))
# country_firms_table[country_firms_table.share_businesses_with_website.isna()].country.head(50)

# plot jaccard matrix of domain overlap between crawls
jaccard_matrix = pd.read_csv('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/data/cc_orbis_global_jaccard_matrix.csv', index_col=0)
jaccard_matrix = jaccard_matrix.astype(float)
fig, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True, center='light')
mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool))
sns.heatmap(jaccard_matrix, mask=mask, cmap=cmap, vmax=1, center=0.5, square=True, linewidths=.5,
            cbar_kws={"shrink": .5}, annot=True, fmt='.2f')
ax.set_xticklabels(fetch_times.dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax.set_yticklabels(fetch_times.dt.strftime('%Y-%m-%d'))
ax.set_xlabel('Crawl date')
ax.set_ylabel('Crawl date')
ax.set_title('Domain overlap (Jaccard similarity) between crawls')
fig.tight_layout()
plt.savefig('C:/Users/Jakob/Documents/SECO_COVID_Data/output 2024/plots/cc_orbis_global_jaccard_similarity_between_crawls.pdf', dpi=300)
plt.show()
