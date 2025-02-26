# Replication package for "Real-time Monitoring of Economic Shocks using Company Websites"

To generate our web-based affectedness indicator (WAI) from CommonCrawl archive data:
  - Run keywords_languages.py to get a keyword list extended with all translations of the original keywords
  - Download: To generate the list of company websites used in our sample, access to ORBIS firm-level data is required. Select companies with at least 5 employees, as well as non-empty fields for city, turnover, and website address at any point between 2015 and now. Given the URL list, follow the instructions at the [CommonCrawl downloader repository](https://github.com/jakob-ra/cc-download). Expected costs are around 200$.
  - Run llm_inference_llama_3_1.ipynb to run LLM inference on Covid-mentioning passages. Expected costs are around 2000$ with current cloud gpu pricing (February 2025).
  - Run read-cc-orbis-global-for-llm.ipynb to combine outputs
    
Plots by country and industry can be replicated with access to ORBIS firm-level data:
  - Run SQL queries in orbis_athena_queries to read ORBIS files as a database and extract the required data
  - Run read_cc_results_orbis_global_llm.py to combine web results with ORBIS data and create the input file for the plotting script
  - Run plot_cc_results_orbis_global_llm.py

Regression tables can be replicated with access to Compustat data (for quarterly stock returns, sales, and other variable):
  - Download files from Compustat Global and North America (2017-2022)
  - Run compustat-quarterly.py to generate input file for regressions
  - Run compustat-full-turnover-vs-cc-results.do
