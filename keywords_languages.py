import pandas as pd
import requests

path = 'https://commoncrawl.github.io/cc-crawl-statistics/plots/languages'
r = requests.get(path)
df = pd.read_html(r.text)[0]
df = df[df.columns[:2]]
df.columns = ['lang', 'share']
df.sort_values('share', ascending=False, inplace=True)
df = df[df.lang != '<unknown>']


# languagee alpha-3 to name
import pycountry

def get_lang_name(x):
    try:
        return pycountry.languages.get(alpha_3=x).name
    except:
        return None

df['lang_name'] = df.lang.apply(lambda x: get_lang_name(x))
df.reset_index(drop=True, inplace=True)
df = df[df.share >= 0.01]

keywords_english = ['corona', 'covid', 'covid-19', 'sars‑cov‑2', 'coronavirus', 'pandemic']

# use gpt 3.5 to translate keywords
import openai

with open('C:/Users/Jakob/Documents/openai-key.txt', 'r') as f:
    openai.api_key = f.read()

translated_keywords = {}
for lang in list(df.lang_name.unique()) + ['Chinese (Simplified)', 'Chinese (Traditional)', 'Chinese (Cantonese)', 'Chinese (Mandarin)']:
    print(lang)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":
        f"Translate the following keywords into {lang}:\n{', '.join(keywords_english)}\n Please only output the translated keywords, each on a new line."}])
    translated_keywords[lang] = response.choices[0].message.content.split('\n')
    print(translated_keywords[lang])

df_keywords = pd.DataFrame([[k, ', '.join(v)] for k,v in translated_keywords.items()], columns=['lang', 'keywords'])
df_keywords['keywords'] = df_keywords.keywords.str.split(', ')

# remove everything in brackets from translated keywords
import re
df_keywords['keywords'] = df_keywords.keywords.apply(lambda keywords: [re.sub(r'\(.*\)', '', keyword).strip().lower() for keyword in keywords])
df_keywords['lang'] = df_keywords.lang.apply(lambda x: re.sub(r'\(.*\)', '', x).strip())
df_keywords = df_keywords.groupby('lang').keywords.apply(sum).reset_index()

# print to latex table
keywords_pretty = df_keywords.copy()
keywords_pretty['keywords'] = keywords_pretty.keywords.apply(lambda x: ', '.join(x))
keywords_pretty.columns = ['Language', 'Keywords']
keywords_pretty.to_latex('C:/Users/Jakob/Downloads/keywords_languages.tex', index=False, escape=False, encoding='utf_32')

unique_keywords = pd.Series(df_keywords.keywords.sum()).drop_duplicates()
unique_keywords.to_csv('C:/Users/Jakob/Downloads/covid_keywords.csv', index=False, header=False)