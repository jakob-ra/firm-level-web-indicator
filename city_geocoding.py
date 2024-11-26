import pandas as pd
import country_converter as coco
from Levenshtein import ratio as levenshtein_distance
from tqdm import tqdm
tqdm.pandas()

def format_match_strings(match_strings: pd.Series):
    match_strings = match_strings.astype(str)
    match_strings = match_strings.str.replace(r'\(.*\)', '', regex=True)  # remove anything in brackets
    match_strings = match_strings.str.lower()
    match_strings = match_strings.str.replace(r'\s+', ' ', regex=True)
    match_strings = match_strings.str.strip()

    return match_strings
def find_best_match(string_to_match: str, candidates: pd.Series, threshold: float,
                    similarity_func=levenshtein_distance):
    if candidates.empty:
        return None
    candidates.reset_index(drop=True, inplace=True)

    # Check for exact match first
    exact_match = candidates[candidates == string_to_match]
    if not exact_match.empty:
        match = exact_match.iloc[0]

        return match  # Exact match found

    # calculate similarity scores
    similarities = candidates.apply(lambda x: similarity_func(string_to_match, x))
    max_similarity = similarities.max()
    max_similarity_index = similarities.idxmax()

    if max_similarity > threshold:
        best_match = candidates.loc[max_similarity_index]
        match = best_match

        return match

    else:
        return None  # No match above threshold


if __name__ == '__main__':
    cities = pd.read_parquet('C:/Users/Jakob/Downloads/covid_orbis_global_cities.parquet')
    df_to_be_matched = cities
    city_column = 'city'
    region_column = 'region_level_1'
    country_column = 'country'
    country_is_iso2 = False
    allow_match_without_region = True
    reduced_alternate_names = True
    similarity_threshold = 0.75

    # create unique row id
    df_to_be_matched['row_id'] = df_to_be_matched.reset_index().index

    match_columns = [x for x in [city_column, region_column, country_column] if x is not None]
    unique_cities = df_to_be_matched.drop_duplicates(subset=match_columns)

    rename_dict = {city_column: 'city', region_column: 'region', country_column: 'country'}
    unique_cities.rename(columns=rename_dict, inplace=True)
    match_columns = [rename_dict[x] for x in match_columns]

    unique_cities = unique_cities[unique_cities.city.str.len() > 1]

    if country_column and not country_is_iso2:
        unique_countries = unique_cities.country.drop_duplicates().to_frame()
        unique_countries['country_iso2'] = coco.convert(names=unique_countries['country'], to='ISO2')
        unique_cities = unique_cities.merge(unique_countries[['country', 'country_iso2']], on='country',
                                            how='left')
        unique_cities['country'] = unique_cities['country_iso2']
        unique_cities.drop(columns='country_iso2', inplace=True)
        unique_cities = unique_cities.explode('country')  # for countries with multiple iso2 codes, create multiple rows

    if region_column:
        region_stop_words = ['Province', 'Department', 'County', 'Region', 'Oblast', 'Governorate',
                             'Municipality', 'District', 'City', 'Town', 'Krai', 'kraj', 'Imarat',
                             'Republic of ', 'Territory', 'Autonomous', 'of']
        unique_cities['region'] = unique_cities['region'].str.replace('|'.join(region_stop_words), '', regex=True, case=False)

    for col in match_columns:
        unique_cities[col] = format_match_strings(unique_cities[col])
    unique_cities.drop_duplicates(subset=match_columns, inplace=True)  # remove alternate spellings of country

    # load geonames admin1 names
    if region_column:
        geonames_admin1_names_url = 'https://github.com/jakob-ra/financial_news/raw/master/geonames_admin1_alternate_names.parquet'
        geonames_admin1_names = pd.read_parquet(geonames_admin1_names_url)
        geonames_admin1_names.rename(columns={'country_code': 'country'}, inplace=True)
        geonames_admin1_names['country'] = format_match_strings(geonames_admin1_names['country'])
        geonames_admin1_names['admin1_name'] = geonames_admin1_names['admin1_name'].str.replace('|'.join(region_stop_words), '', regex=True, case=False)
        geonames_admin1_names['admin1_name'] = format_match_strings(geonames_admin1_names['admin1_name'])
        geonames_admin1_names['admin1_code'] = format_match_strings(geonames_admin1_names['admin1_code'])
        geonames_admin1_names.drop_duplicates(subset=['admin1_name', 'country'], inplace=True)
        geonames_admin1_names = geonames_admin1_names[geonames_admin1_names['admin1_name'].str.len() > 1]

    # load geonames cities
    geonames_cities_500_path = 'http://download.geonames.org/export/dump/cities500.zip'

    geonames = pd.read_csv(geonames_cities_500_path, compression='zip', low_memory=False, sep='\t', header=None,
                           names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
                                  'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
                                  'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem',
                                  'timezone', 'modification_date'])
    geonames.sort_values(by='population', ascending=False, inplace=True)
    geonames.rename(columns={'country_code': 'country'}, inplace=True)

    cities[cities.country=='Slovenia'].groupby('region_level_1').firm_count.sum().sort_values(ascending=False).head(10)

    if reduced_alternate_names:
        geonames['city'] = geonames.apply(
                lambda row: [row['asciiname']] + [x for x in str(row.alternatenames).split(',') if
                                                               not x == 'nan'][:2], axis=1)
    else:
        geonames['city'] = geonames.apply(
                lambda row: [row['name'], row['asciiname']] + [x for x in str(row.alternatenames).split(',') if
                                                               not x == 'nan'], axis=1)
    geonames = geonames.explode('city')

    match_columns_geonames = ['city']
    if region_column:
        match_columns_geonames.append('admin1_code')
    if country_column:
        match_columns_geonames.append('country')

    for col in match_columns_geonames:
        geonames[col] = format_match_strings(geonames[col])

    geonames['match_name'] = geonames[match_columns_geonames].astype(str).apply(lambda x: ', '.join(x), axis=1)
    geonames.drop_duplicates('match_name', inplace=True)

    relevant_columns = match_columns_geonames + ['latitude', 'longitude', 'population']
    geonames = geonames[relevant_columns]

    geonames = geonames[geonames.city.str.len() > 1]


    # match
    unique_cities['match_name'] = unique_cities[match_columns].astype(str).apply(lambda x: ', '.join(x), axis=1)

    # match regions
    if region_column:
        unique_regions = unique_cities[['region', 'country']].drop_duplicates()
        unique_regions = unique_regions[unique_regions['region'].str.len() > 1]
        unique_regions['admin1_name'] = unique_regions.progress_apply(
                lambda row: find_best_match(row['region'], geonames_admin1_names[
                    geonames_admin1_names['country'] == row['country']]['admin1_name'],
                                                 similarity_threshold), axis=1)
        unique_regions = unique_regions.merge(geonames_admin1_names[['admin1_name', 'country', 'admin1_code']],
                                          on=['admin1_name', 'country'], how='left')
        unique_cities = unique_cities.merge(unique_regions, on=['region', 'country'], how='left')

        unique_cities.sort_values('firm_count', ascending=False, inplace=True)

    # Find matches
    if country_column and region_column:
        unique_cities['city_matched'] = unique_cities.progress_apply(
                lambda row: find_best_match(row['city'], geonames[
                    (geonames['country'] == row['country']) & (
                                geonames['admin1_code'] == row['admin1_code'])]['city'],
                                            similarity_threshold), axis=1)
        if allow_match_without_region:
            unique_cities.loc[unique_cities['city_matched'].isna(), 'city_matched'] = unique_cities.loc[
                unique_cities['city_matched'].isna()].progress_apply(
                    lambda row: find_best_match(row['city'],
                                                geonames[geonames['country'] == row['country']]['city'],
                                                similarity_threshold), axis=1)

    elif country_column and not region_column:
        unique_cities['city_matched'] = unique_cities.progress_apply(
                lambda row: find_best_match(row['city'], geonames[
                    geonames['country'] == row['country']]['city'],
                                                 similarity_threshold), axis=1)
    else:
        unique_cities['city_matched'] = unique_cities['city'].progress_apply(
                lambda row: find_best_match(row, geonames['city'], similarity_threshold))

    # Merge geonames data to unique_cities
    unique_cities = unique_cities.merge(geonames.rename(columns={'city': 'city_matched'}),
                                        on=['city_matched', 'admin1_code', 'country'], how='left')

    # Drop unnecessary columns
    unique_cities.drop(columns=['match_name'], inplace=True)

    unique_cities.to_parquet('C:/Users/Jakob/Downloads/covid_orbis_global_cities_geocoded.parquet')

    # manual matching
    geonames_cols = ['latitude', 'longitude', 'population']
    unique_cities.loc[unique_cities.city == 'roma', geonames_cols] = geonames.loc[geonames.city == 'rome', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'wien', geonames_cols] = geonames.loc[geonames.city == 'vienna', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'sofia', geonames_cols] = geonames.loc[geonames.city == 'sofia', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'muenchen', geonames_cols] = geonames.loc[geonames.city == 'munich', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'lodz', geonames_cols] = geonames.loc[geonames.city == 'lodz', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'kiev', geonames_cols] = geonames.loc[geonames.city == 'kyiv', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'zagreb', geonames_cols] = geonames.loc[geonames.city == 'zagreb', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'dubai', geonames_cols] = geonames.loc[geonames.city == 'dubai', geonames_cols].iloc[0].values
    unique_cities.loc[unique_cities.city == 'napoli', geonames_cols] = geonames.loc[geonames.city == 'naples', geonames_cols].iloc[0].values

    # fill na with lat long of biggest city in admin1
    unique_cities[geonames_cols] = unique_cities[geonames_cols].fillna(
        unique_cities.drop(columns=geonames_cols).merge(
            unique_cities.groupby(['country', 'admin1_code'])[geonames_cols].first().reset_index(),
            on=['country', 'admin1_code'], how='left'))

    unique_cities.to_parquet('C:/Users/Jakob/Downloads/covid_orbis_global_cities_geocoded_fillna.parquet')

    cities_matched = cities.merge(unique_cities[['row_id', 'admin1_name', 'admin1_code', 'city_matched', 'latitude', 'longitude', 'population']],
                                  on=['row_id'], how='left')
    cities_matched.drop(columns=['row_id'], inplace=True)
    cities_matched.to_parquet('C:/Users/Jakob/Downloads/covid_orbis_global_cities_matched_geocoded_fillna.parquet')











    df_cities = pd.DataFrame({'city'   : ['Berlin', 'Hamburg', 'Munich', 'Berlin', 'Paris', 'Paris'],
                              'region' : ['Berlin', 'Hamburg', 'Bavaria', 'Berlin', 'Ile-de-France', 'Texas'],
                              'country': ['Germany', 'Germany', 'Germany', 'Germany', 'France', 'USA']})
    city_country_aggregates = pd.read_pickle('C:/Users/Jakob/Downloads/city_country_aggregates.pkl')
    cg = CityGeocoder(cities.sample(10),
                      city_column='city',
                      region_column='region_level_1',
                      country_column='country',
                      similarity_threshold=0.5,
                      cached_geonames_path='/Users/Jakob/Downloads/cities500.zip')
    df_cities_geocoded = cg.geocode_cities()


    # run code below to get variants of admin1 names
    alternate_names = pd.read_csv('/Users/Jakob/Downloads/alternateNamesV2/alternateNamesV2.txt', sep='\t',
                header=None, names=['alternateNameId', 'geonameid', 'isolanguage', 'alternate_name', 'isPreferredName',
                                    'isShortName', 'isColloquial', 'isHistoric', 'from', 'to'], low_memory=False,
                                  usecols=['geonameid', 'alternate_name'])

    geonames_admin1_names_url = 'https://download.geonames.org/export/dump/admin1CodesASCII.txt'
    geonames_admin1_names = pd.read_csv(geonames_admin1_names_url, sep='\t', header=None,
                                        names=['code', 'name', 'name_ascii', 'geonameid'],
                                        usecols=['geonameid', 'code'])
    geonames_admin1_names = geonames_admin1_names.merge(alternate_names, on='geonameid', how='left')
    geonames_admin1_names.drop_duplicates(inplace=True)
    geonames_admin1_names['country_code'] = geonames_admin1_names.code.str.split('.').str[0]
    geonames_admin1_names['admin1_code'] = geonames_admin1_names.code.str.split('.').str[1]
    geonames_admin1_names.drop(columns=['code'], inplace=True)
    geonames_admin1_names.rename(columns={'alternate_name': 'admin1_name'}, inplace=True)
    geonames_admin1_names.to_parquet('/Users/Jakob/Downloads/geonames_admin1_alternate_names.parquet')

    # ## run code below to get variants of admin1 names
    #
    # # gadm_admin1_path = '/Users/Jakob/Downloads/gadm_admin1_names.csv'
    # gadm_admin1_path = 'https://docs.google.com/spreadsheets/d/1S0_Wl0bM8EAyX23M4Yld7nIuh6esoWh9IVs_QrWYeBU/gviz/tq?tqx=out:csv&sheet=gadm36_1'
    # gadm_admin1 = pd.read_csv(gadm_admin1_path)
    # gadm_admin1['country_iso2'] = coco.convert(names=gadm_admin1['GID_0_0'], to='ISO2')
    # gadm_admin1 = gadm_admin1.explode('country_iso2')
    # gadm_admin1['gadm_all_names'] = gadm_admin1['NAME_1']
    # gadm_admin1.loc[gadm_admin1['VARNAME_1'].notna(), 'gadm_all_names'] = gadm_admin1.loc[gadm_admin1['VARNAME_1'].notna(), 'NAME_1'].astype(str) + '|' + gadm_admin1.loc[gadm_admin1['VARNAME_1'].notna(), 'VARNAME_1'].astype(str)
    # gadm_admin1['match_name'] = gadm_admin1['gadm_all_names'].str.split('|')
    # gadm_admin1 = gadm_admin1.explode('match_name')
    # gadm_admin1['match_name'] = gadm_admin1['gadm_all_names'].str.split('|')
    # gadm_admin1 = gadm_admin1.explode('match_name')
    # gadm_admin1.drop_duplicates(subset=['match_name', 'country_iso2'], inplace=True)
    #
    #
    # geonames_admin1_names_url = 'https://download.geonames.org/export/dump/admin1CodesASCII.txt'
    # geonames_admin1_names = pd.read_csv(geonames_admin1_names_url, sep='\t', header=None,
    #                                     names=['code', 'name', 'name_ascii', 'geonameid'])
    # geonames_admin1_names['country_code'] = geonames_admin1_names.code.str.split('.').str[0]
    #
    # geonames_admin1_names['admin1_code'] = geonames_admin1_names.code.str.split('.').str[1]
    # geonames_admin1_names.drop(columns=['code', 'geonameid'], inplace=True)
    # geonames_admin1_names.rename(columns={'name': 'admin1_name', 'name_ascii': 'admin1_name_ascii'},
    #                              inplace=True)
    #
    # stop_words = ['Province', 'Department', 'County', 'Region', 'Oblast', 'Governorate', 'Municipality',
    #               'District', 'Republic', 'City', 'Town', 'Krai', 'kraj', 'Imarat', 'Republic of ']
    # geonames_admin1_names['admin1_name_alt'] = geonames_admin1_names['admin1_name'].str.replace('|'.join(stop_words), '', regex=True)
    # geonames_admin1_names['admin1_name_alt'] = geonames_admin1_names['admin1_name_alt'].str.replace(r'\s+', ' ', regex=True)
    # geonames_admin1_names['admin1_name_ascii_alt'] = geonames_admin1_names['admin1_name_ascii'].str.replace('|'.join(stop_words), '', regex=True)
    # geonames_admin1_names['admin1_name_ascii_alt'] = geonames_admin1_names['admin1_name_ascii_alt'].str.replace(r'\s+', ' ', regex=True)
    # geonames_admin1_names['geonames_all_names'] = geonames_admin1_names['admin1_name'].astype(str) + '|' + geonames_admin1_names['admin1_name_ascii'].astype(str) + '|' + geonames_admin1_names['admin1_name_alt'].astype(str) + '|' + geonames_admin1_names['admin1_name_ascii_alt'].astype(str)
    # geonames_admin1_names['match_name'] = geonames_admin1_names['geonames_all_names'].str.split('|')
    # geonames_admin1_names = geonames_admin1_names.explode('match_name')
    # geonames_admin1_names.drop_duplicates(subset=['match_name', 'country_code'], inplace=True)
    #
    # gadm_admin1['match_name'] = gadm_admin1['match_name'].astype(str) + ', ' + gadm_admin1['country_iso2'].astype(str)
    # gadm_admin1['match_name'] = cg.format_match_strings(gadm_admin1['match_name'])
    # geonames_admin1_names['match_name'] = geonames_admin1_names['match_name'].astype(str) + ', ' + geonames_admin1_names['country_code'].astype(str)
    # geonames_admin1_names['match_name'] = cg.format_match_strings(geonames_admin1_names['match_name'])
    #
    # best_matches = geonames_admin1_names['match_name'].progress_apply(lambda x: cg.find_best_match(x, gadm_admin1['match_name'], 0.8))
    # best_matches = pd.DataFrame(best_matches.tolist(), columns=['match_name', 'gadm_match_name', 'similarity'])
    #
    # geonames_admin1_names = geonames_admin1_names.merge(best_matches, on='match_name', how='left')
    #
    # geonames_admin1_names = geonames_admin1_names.merge(gadm_admin1, left_on='gadm_match_name',
    #                                                     right_on='match_name', how='left')
    #
    # geonames_admin1_names['all_names'] = geonames_admin1_names['geonames_all_names'].astype(str) + '|' + geonames_admin1_names['gadm_all_names'].astype(str)
    # geonames_admin1_names['all_names'] = geonames_admin1_names['all_names'].str.split('|').apply(lambda x: list(set([y for y in x if y not in ['', 'nan']])))
    # geonames_admin1_names['all_names'] = geonames_admin1_names['all_names'].apply(lambda x: list(set([y for y in x if y not in ['', 'nan']])))
    #
    # keep_cols = ['admin1_name', 'country_code', 'admin1_code', 'LAT', 'LON', 'all_names']
    # geonames_admin1_names = geonames_admin1_names[keep_cols].copy(deep=True)
    # geonames_admin1_names = geonames_admin1_names.groupby(['country_code', 'admin1_code', 'admin1_name']).agg({'LAT': 'first', 'LON': 'first', 'all_names': 'sum'})
    # geonames_admin1_names['all_names'] = geonames_admin1_names['all_names'].apply(lambda x: list(set(x)))
    # geonames_admin1_names.reset_index(inplace=True)
    #
    # geonames_admin1_names.to_parquet('/Users/Jakob/Downloads/geonames_admin1_alternate_names.parquet')



class CityGeocoder:
    """Geocodes the cities in df_to_be_matched with latitude and longitude"""

    def __init__(self, df_to_be_matched: pd.DataFrame, city_column: str = 'city', region_column=None,
                 country_column=None, geonames_lookup=True, api_lookup=False, country_is_iso2=False,
                 cached_geonames_path=None, similarity_threshold=0.8, allow_match_without_region=True):
        if city_column not in df_to_be_matched.columns:
            raise ValueError(f'Column {city_column} not in DataFrame')
        if region_column and region_column not in df_to_be_matched.columns:
            raise ValueError(f'Column {region_column} not in DataFrame')
        if country_column and country_column not in df_to_be_matched.columns:
            raise ValueError(f'Column {country_column} not in DataFrame')
        if region_column and not country_column:
            raise ValueError(f'If region_column is given, country_column must be given as well.')

        self.df_to_be_matched = df_to_be_matched
        self.city_column = city_column
        self.region_column = region_column
        self.country_column = country_column
        self.match_columns = [x for x in [city_column, region_column, country_column] if x is not None]
        self.geonames_lookup = geonames_lookup
        self.api_lookup = api_lookup
        self.country_is_iso2 = country_is_iso2
        self.cached_geonames_path = cached_geonames_path
        self.similarity_threshold = similarity_threshold
        self.allow_match_without_region = allow_match_without_region

        self.unique_cities = self.prepare_df_unique_cities()

    def prepare_df_unique_cities(self):
        unique_cities = self.df_to_be_matched.drop_duplicates(subset=self.match_columns)
        unique_cities = unique_cities[unique_cities.city.str.len() > 1]

        if self.country_column:
            unique_countries = self.df_to_be_matched.country.drop_duplicates().to_frame()

            # country to iso2
            if not self.country_is_iso2:
                unique_countries['country_iso2'] = coco.convert(names=unique_countries[self.country_column], to='ISO2')
                unique_cities = unique_cities.merge(unique_countries[['country', 'country_iso2']],
                                                    on='country', how='left')
                unique_cities[self.country_column] = unique_cities['country_iso2']
                unique_cities.drop(columns='country_iso2', inplace=True)
                unique_cities = unique_cities.explode(self.country_column)  # for countries with multiple iso2 codes, create multiple rows
                unique_cities.drop_duplicates(subset=self.match_columns,
                                              inplace=True)  # remove alternate spellings of country

        return unique_cities

    def get_geonames_admin1_names(self):
        geonames_admin1_names_url = 'https://github.com/jakob-ra/financial_news/raw/master/geonames_admin1_alternate_names.parquet'
        geonames_admin1_names = pd.read_parquet(geonames_admin1_names_url)

        return geonames_admin1_names

    def get_geonames_cities_500(self):
        if self.cached_geonames_path:
            geonames_cities_500_path = self.cached_geonames_path
        else:
            geonames_cities_500_path = 'http://download.geonames.org/export/dump/cities500.zip'

        geonames = pd.read_csv(geonames_cities_500_path, compression='zip', low_memory=False, sep='\t',
                               header=None,
                               names=['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude',
                                      'longitude', 'feature_class', 'feature_code', 'country_code', 'cc2',
                                      'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
                                      'population', 'elevation', 'dem', 'timezone', 'modification_date'])
        geonames.sort_values(by='population', ascending=False, inplace=True)

        geonames['city'] = geonames.apply(
                lambda row: [row['name'], row['asciiname']] + [x for x in str(row.alternatenames).split(',')
                                                               if not x == 'nan'], axis=1)
        geonames = geonames.explode('city')

        match_cols = ['city']
        if self.region_column:
            match_cols.append('admin1_code')
        if self.country_column:
            match_cols.append('country_code')
        geonames['match_name'] = ', '.join(geonames[match_cols].astype(str), axis=1)
        geonames.drop_duplicates(subset='match_name', inplace=True)

        relevant_columns = match_cols + ['latitude', 'longitude', 'population']
        geonames = geonames[relevant_columns]

        geonames = geonames[geonames.city.str.len() > 1]

        return geonames

    @staticmethod
    def format_match_strings(match_strings: pd.Series):
        match_strings = match_strings.astype(str)
        match_strings = match_strings.str.replace(r'\(.*\)', '', regex=True) # remove anything in brackets
        match_strings = match_strings.str.lower()
        match_strings = match_strings.str.replace(r'\s+', ' ', regex=True)
        match_strings = match_strings.str.strip()

        return match_strings

    @staticmethod
    def find_best_match(string_to_match: str, candidates: pd.Series, threshold: float, similarity_func=levenshtein_distance):
        if candidates.empty:
            return string_to_match, None, 0.0
        candidates.reset_index(drop=True, inplace=True)

        # Check for exact match first
        exact_match = candidates[candidates == string_to_match]
        if not exact_match.empty:
            match = exact_match.iloc[0]

            return match # string_to_match, match, 1.0  # Exact match found

        # calculate similarity scores
        similarities = candidates.apply(lambda x: similarity_func(string_to_match, x))
        max_similarity = similarities.max()
        max_similarity_index = similarities.idxmax()

        if max_similarity > threshold:
            best_match = candidates.loc[max_similarity_index]
            match = best_match

            return match # string_to_match, match, max_similarity

        else:
            return None # string_to_match, None, max_similarity  # No match above threshold

    def match_cities_geonames(self):
        unique_cities_matched = self.unique_cities.copy(deep=True)

        unique_cities_matched['match_name'] = ', '.join(unique_cities_matched[self.match_columns].astype(str), axis=1)

        print('Unique cities prepared for matching')

        # match regions
        if self.region_column:
            unique_regions = self.unique_cities[[self.region_column, self.country_column]].drop_duplicates()
            geonames_admin1_names = self.get_geonames_admin1_names()
            best_matches = unique_regions.progress_apply(
                lambda row: self.find_best_match(row[self.region_column],
                                                 geonames_admin1_names[geonames_admin1_names['country_code'] == row[self.country_column]]['admin1_name'], self.similarity_threshold), axis=1)
            best_matches = pd.DataFrame(best_matches.tolist(), columns=[self.region_column, 'geonames_admin1_name', 'similarity'])
            # replace admin1 name with admin1 code
            best_matches = best_matches.merge(geonames_admin1_names, left_on=['geonames_admin1_name', self.country_column], right_on=['admin1_name', 'country_code'], how='left')
            best_matches = best_matches[[self.region_column, 'admin1_code', 'similarity']]
            unique_cities_matched = unique_cities_matched.merge(best_matches, left_on=[self.region_column, self.country_column], right_on=['admin1_name', 'country_code'], how='left')

            if self.allow_match_without_region:
                unique_cities_matched.loc[unique_cities_matched['geonames_admin1_name'].isna(), 'geonames_admin1_name'] = unique_cities_matched.loc[unique_cities_matched['geonames_admin1_name'].isna(), self.country_column]

        # Find matches
        if self.country_column and self.region_column:
            best_matches = unique_cities_matched.progress_apply(
                lambda row: self.find_best_match(row[self.city_column],
                                                self.geonames[(self.geonames['country'] == row[self.country_column]) & (self.geonames['admin1_name'] == row[self.region_column])]['city'],
                                                self.similarity_threshold), axis=1)
        elif self.country_column and not self.region_column:
            best_matches = unique_cities_matched.progress_apply(
                lambda row: self.find_best_match(row[self.city_column],
                                                self.geonames[self.geonames['country'] == row[self.country_column]]['city'],
                                                self.similarity_threshold), axis=1)
        else:
            best_matches = unique_cities_matched[self.city_column].progress_apply(
                lambda row: self.find_best_match(row, self.geonames['city'], self.similarity_threshold))

        best_matches = pd.DataFrame(best_matches.tolist(), columns=['match_name', 'geonames_match_name', 'similarity'])

        # Merge best matches back to unique_cities
        unique_cities_matched = unique_cities_matched.merge(best_matches, on='match_name', how='left')
        unique_cities_matched.drop(columns=['match_name'], inplace=True)

        # Merge geonames data to unique_cities
        unique_cities_matched = unique_cities_matched.merge(self.geonames, left_on='geonames_match_name', right_on='match_name', how='left')

        # Drop unnecessary columns
        unique_cities_matched.drop(columns=['geonames_match_name'], inplace=True)

        return unique_cities_matched

    def geocode_cities(self):
        if self.geonames_lookup:
            self.geonames = self.get_geonames_cities_500()
            self.unique_cities = self.match_cities_geonames()
        if self.api_lookup:
            pass

        return self.unique_cities