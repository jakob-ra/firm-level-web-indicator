# Replication package for "A Web-Based Indicator for Real-Time Monitoring of Economic Shocks"

To generate our web-based affectedness indicator (WAI) from CommonCrawl archive data:
  - Run keywords_languages.py to get a keyword list extended with all translations of the original keywords
  - Download: TODO
  - Run llm_inference_llama_3_1.ipynb to run LLM inference on Covid-mentioning passages
  - Run read-cc-orbis-global-for-llm.ipynb to combine outputs 

Plots by country and industry can be replicated with access to ORBIS firm-level data:
  - Run SQL queries in orbis_athena_queries to read ORBIS files as a database and extract the required data
  - Run read_cc_results_orbis_global_llm.py to combine web results with ORBIS data and create the input file for the plotting script
  - Run plot_cc_results_orbis_global_llm.py

Regression tables can be replicated with access to Compustat data (for quarterly stock returns, sales, and other variable):
  - Download files from Compustat Global and North America (2017-2022)
  - Run compustat-quarterly.py to generate input file for regressions
  - Run compustat-full-turnover-vs-cc-results.do
