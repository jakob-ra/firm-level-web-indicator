import requests
import pandas as pd
from bs4 import BeautifulSoup
import statsmodels.formula.api as smf
from yahooquery import Ticker
from tqdm import tqdm
import country_converter as coco

def get_nace_code_descriptions():
    link = 'https://ec.europa.eu/competition/mergers/cases/index/nace_all.html'
    page = requests.get(link)
    nace = BeautifulSoup(page.content, 'html.parser')
    nace = str(nace)
    nace = nace.split('2010-03-25<br/><br/>\n<br/>\n\t')[1]
    nace = nace.split('\n<div class="footer"')[0]
    nace = nace.split(' \n\t<br/>\n\t')
    nace = pd.DataFrame(nace, columns=['description'])
    nace['level_full'] = nace.description.apply(lambda x: x.split(' - ')[0])
    nace['description'] = nace.description.apply(lambda x: x.split(' - ')[1])
    nace['level_1'] = nace.level_full.apply(lambda x: x[0])
    nace['level_2'] = nace.level_full.apply(lambda x: x[1:].split('.')[0])
    nace = nace[~nace.level_full.str.contains('\.')]
    nace_sections = nace[nace.level_2=='']
    nace_sections = nace.drop_duplicates('level_2')[['level_1', 'level_2']].merge(nace_sections, how='inner', on='level_1').set_index('level_2_x').description
    nace_sections_dict = nace_sections.iloc[1:].to_dict()
    # take only 2 digit codes
    nace = nace[~nace.level_full.str.contains('\.')]
    nace = nace[nace.level_2.str.len()>0]
    nace = nace[['description', 'level_2']]
    nace.columns = ['nace_description', 'nace_2_digit']
    nace['nace_2_digit'] = nace['nace_2_digit'].astype(int)

    return nace, nace_sections_dict

def remove_url_prefix(url):
    url = url.lower()
    url = url.replace('https://', '')
    url = url.replace('http://', '')
    url = url.replace('www.', '')
    url = url.split('/')[0]
    url = url.replace(',', '')
    # url = url.replace('-', '') don't do this one, as it might be part of the domain name
    url = url.strip()
    if url.startswith('.'):
        url = url[1:]

    return url

def statsmodels_summary_to_df(summary):
    """Converts a statsmodels summary to a pandas dataframe."""
    summary_as_html = summary.tables[1].as_html()
    res = pd.read_html(summary_as_html, header=0, index_col=0)[0]

    return res

def ols_short_summary(res, vars_of_interest, dependent_var, explanatory_var, controls, fixed_effects):
    summary = res.summary()
    try:
        short_summary = statsmodels_summary_to_df(summary).loc[vars_of_interest][['coef', 'std err', 'P>|z|']]
        # add number of observations to short summary
        n = pd.read_html(summary.tables[0].as_html(), header=0, index_col=0)[0].loc['No. Observations:'][0]
        short_summary['n'] = n
    except KeyError:
        print(f'{vars_of_interest} not in summary')
        return None
    short_summary.rename(columns={'P>|z|': 'p_value', 'strd err': 'std_err'}, inplace=True)
    short_summary.reset_index(drop=True, inplace=True)
    short_summary['dependent_var'] = dependent_var
    short_summary['explanatory_var'] = explanatory_var
    short_summary['controls'] = ', '.join(controls)
    short_summary['fixed_effects'] = ', '.join(fixed_effects)

    return short_summary

def remove_dashes_spaces_parentheses(string):
    return string.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')


def ols_regression(data, dependent_var, explanatory_vars, controls=[], fixed_effects=[], normalize=False,
                   return_short_summary=False, sample_selection=None):
    data = data.dropna(subset = [dependent_var] + explanatory_vars + controls + fixed_effects)
    if sample_selection:
        data = sample_selection(data)
    # remove dashes, spaces, and parentheses from column names
    data.columns = [remove_dashes_spaces_parentheses(col) for col in data.columns]
    dependent_var = remove_dashes_spaces_parentheses(dependent_var)
    explanatory_vars = [remove_dashes_spaces_parentheses(col) for col in explanatory_vars]
    controls = [remove_dashes_spaces_parentheses(col) for col in controls]
    fixed_effects = [remove_dashes_spaces_parentheses(col) for col in fixed_effects]
    for var in [dependent_var] + explanatory_vars:
        data[var] = data[var].astype(float)
    if normalize:
        for var in [dependent_var] + controls:
            data[var] = (data[var] - data[var].mean()) / data[var].std()
    formula = f'{dependent_var} ~ {" + ".join(explanatory_vars)}'
    if len(controls) > 0:
        formula += f' + {" + ".join(controls)}'
    if len(fixed_effects) > 0:
        formula += ''.join([f' + C({fe})' for fe in fixed_effects])
    reg = smf.ols(formula=formula, data=data)
    res = reg.fit(cov_type='HC1')

    if len(explanatory_vars) == 1 and data[explanatory_vars[0]].dropna().unique().all() in [0, 1]:
        res.dummy_count = str(data[data[explanatory_vars[0]] == 1].shape[0])
    else:
        res.dummy_count = '-'

    if return_short_summary:
        short_summary = ols_short_summary(res, vars_of_interest=explanatory_vars, dependent_var=dependent_var,
                                          explanatory_var=explanatory_vars[0], controls=controls,
                                          fixed_effects=fixed_effects)
        return short_summary
    else:
        return res

def get_currency_conversion_rates(currencies, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='D').astype(str).tolist()
    conversion_rates = []
    for currency in tqdm(currencies):
        try:
            currency_data = pd.DataFrame(index=dates).reset_index()
            currency_data['symbol'] = currency
            currency_data.rename(columns={'index': 'date'}, inplace=True)
            ticker = Ticker(currency + '=X')
            data = ticker.history(start=start_date, end=end_date)
            data = data['close'].reset_index()
            data['date'] = data['date'].astype(str)
            data.drop(columns=['symbol'], inplace=True)
            currency_data = currency_data.merge(data, how='left', on='date', validate='1:1')
            conversion_rates.append(currency_data)
        except Exception as e:
            print(e)
            print(currency) # georgian lari and zimbabwean dollar not available on yahoo finance
            pass
    conversion_rates = pd.concat(conversion_rates, axis=0, ignore_index=True)
    conversion_rates.sort_values(['symbol', 'date'], inplace=True)
    conversion_rates['close'] = conversion_rates.groupby('symbol')['close'].ffill()

    return conversion_rates

def get_nace_to_wk08_dict(all_nace):
    nace_to_wk08 = pd.read_excel('C:/Users/Jakob/Documents/SECO_COVID_Data/Definition WK08.xls',
                                 sheet_name='wk08_def_export', skiprows=2,
                                 usecols=['NOGA 2008', 'Bezeichnung (EN)']).dropna()
    nace_to_wk08.columns = ['nace', 'wk08']
    nace_to_wk08['nace'] = nace_to_wk08.nace.apply(str).str.split(',')
    nace_to_wk08 = nace_to_wk08.explode('nace')
    nace_to_wk08_dict = {}
    for nace_1 in nace_to_wk08.nace.unique():
        for nace_2 in all_nace:
            if str(nace_2).startswith(nace_1):
                nace_to_wk08_dict[nace_2] = nace_to_wk08.loc[nace_to_wk08.nace == nace_1, 'wk08'].values[0]

    return nace_to_wk08_dict


def get_country_codes_conversion_dicts():
    country_codes_url = 'https://www.iban.com/country-codes'
    country_codes = pd.read_html(country_codes_url)[0]

    country_codes_2_to_3_digit_dict = country_codes.set_index('Alpha-2 code')['Alpha-3 code'].to_dict()
    country_codes_3_to_2_digit_dict = country_codes.set_index('Alpha-3 code')['Alpha-2 code'].to_dict()

    return country_codes_2_to_3_digit_dict, country_codes_3_to_2_digit_dict

def country_codes_conversion(country_codes: pd.Series, source='alpha-2', target='alpha-3'):
    country_codes_2_to_3_digit_dict, country_codes_3_to_2_digit_dict = get_country_codes_conversion_dicts()

    if source == 'alpha-2' and target == 'alpha-3':
        country_codes = country_codes.map(country_codes_2_to_3_digit_dict)
    elif source == 'alpha-3' and target == 'alpha-2':
        country_codes = country_codes.map(country_codes_3_to_2_digit_dict)
    else:
        raise ValueError('source and target must be in ["alpha-2", "alpha-3"]')

    return country_codes

def get_oxford_policy_tracker(aggregation_period='M', extended_cols=False, save_path=None,
                              country_codes='alpha-3'):
    # read data
    oxford_policy_tracker_url = 'https://github.com/OxCGRT/covid-policy-tracker/blob/master/data' \
                                '/OxCGRT_nat_latest.csv?raw=true'
    oxford_cols = ['CountryCode', 'Date', 'StringencyIndex_Average', 'EconomicSupportIndex']
    if extended_cols:
        oxford_cols += ['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport',
                        'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief',
                        'E3_Fiscal measures', 'E4_International support', 'PopulationVaccinated',
                        'StringencyIndex_Average', 'StringencyIndex_Average_ForDisplay',
                        'GovernmentResponseIndex_Average', 'ContainmentHealthIndex_Average',
                        'EconomicSupportIndex']
    oxford_policy_tracker = pd.read_csv(oxford_policy_tracker_url, usecols=oxford_cols)

    oxford_policy_tracker['Date'] = pd.to_datetime(oxford_policy_tracker['Date'], format='%Y%m%d')
    if aggregation_period:
        oxford_policy_tracker['Date'] = oxford_policy_tracker['Date'].dt.to_period(aggregation_period)
        old = oxford_policy_tracker.copy()
        oxford_policy_tracker = oxford_policy_tracker.groupby(['CountryCode', 'Date']).mean()
        # oxford_policy_tracker['E3_Fiscal measures'] = old.groupby(['CountryCode', 'Date'])['E3_Fiscal measures'].sum()
        oxford_policy_tracker = oxford_policy_tracker.reset_index()

    # normalize cases and deaths by population
    covid_data_cols = ['iso_code', 'date', 'new_cases_per_million', 'new_deaths_per_million']
    covid_data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',
                             usecols=covid_data_cols)
    covid_data['date'] = pd.to_datetime(covid_data['date'], format='%Y-%m-%d')
    covid_data['date'] = covid_data['date'].dt.to_period(aggregation_period)
    covid_data = covid_data.groupby(['iso_code', 'date']).mean().reset_index()
    covid_data = covid_data.rename(columns={'iso_code': 'CountryCode', 'date': 'Date'})
    oxford_policy_tracker = pd.merge(oxford_policy_tracker, covid_data, on=['CountryCode', 'Date'],
                                     how='left')

    if country_codes == 'alpha-2':
        oxford_policy_tracker['CountryCode'] = country_codes_conversion(oxford_policy_tracker['CountryCode'],
                                                                        source='alpha-3', target='alpha-2')

    if save_path:
        oxford_policy_tracker.to_csv(save_path, index=False)

    return oxford_policy_tracker


def get_oxford_policy_tracker_regional(country, aggregation_period='M', extended_cols=False, save_path=None,
                              country_codes='alpha-3'):
    assert country in ['Australia', 'Brazil', 'Canada', 'China', 'India', 'United Kingdom', 'United States'], \
    'Country must be one of [Australia, Brazil, Canada, China, India, United Kingdom, United States]'
    # read data
    oxford_policy_tracker_url = f'https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/{country}'
    oxford_policy_tracker_url += f'/OxCGRT_{coco.convert(names=[country], to="ISO3")}_latest.csv?raw=true'
    # replace spaces
    oxford_policy_tracker_url = oxford_policy_tracker_url.replace(' ', '%20')
    oxford_cols = ['RegionName', 'Date', 'StringencyIndex_Average', 'EconomicSupportIndex']
    if extended_cols:
        oxford_cols += ['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport',
                        'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief',
                        'E3_Fiscal measures', 'E4_International support', 'PopulationVaccinated',
                        'StringencyIndex_Average', 'StringencyIndex_Average_ForDisplay',
                        'GovernmentResponseIndex_Average', 'ContainmentHealthIndex_Average',
                        'EconomicSupportIndex']
    oxford_policy_tracker = pd.read_csv(oxford_policy_tracker_url, usecols=oxford_cols)

    oxford_policy_tracker['Date'] = pd.to_datetime(oxford_policy_tracker['Date'], format='%Y%m%d')
    if aggregation_period:
        oxford_policy_tracker['Date'] = oxford_policy_tracker['Date'].dt.to_period(aggregation_period)
        old = oxford_policy_tracker.copy()
        oxford_policy_tracker = oxford_policy_tracker.groupby(['RegionName', 'Date']).mean()
        # oxford_policy_tracker['E3_Fiscal measures'] = old.groupby(['CountryCode', 'Date'])['E3_Fiscal measures'].sum()
        oxford_policy_tracker = oxford_policy_tracker.reset_index()

    # normalize cases and deaths by population
    # covid_data_cols = ['iso_code', 'date', 'new_cases_per_million', 'new_deaths_per_million']
    # covid_data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',
    #                          usecols=covid_data_cols)
    # covid_data['date'] = pd.to_datetime(covid_data['date'], format='%Y-%m-%d')
    # covid_data['date'] = covid_data['date'].dt.to_period(aggregation_period)
    # covid_data = covid_data.groupby(['iso_code', 'date']).mean().reset_index()
    # covid_data = covid_data.rename(columns={'iso_code': 'CountryCode', 'date': 'Date'})
    # oxford_policy_tracker = pd.merge(oxford_policy_tracker, covid_data, on=['CountryCode', 'Date'],
    #                                  how='left')
    #
    # if country_codes == 'alpha-2':
    #     oxford_policy_tracker['CountryCode'] = country_codes_conversion(oxford_policy_tracker['CountryCode'],
    #                                                                     source='alpha-3', target='alpha-2')
    #
    # if save_path:
    #     oxford_policy_tracker.to_csv(save_path, index=False)

    return oxford_policy_tracker