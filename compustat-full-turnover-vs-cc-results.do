clear
est clear

use "C:\Users\Jakob\Downloads\reg_df_29-10-24.dta"

encode url, gen(url1)
encode nace_section, gen(nace_section1)
encode wk08, gen(wk081)
encode loc, gen(loc1)
encode Country_Region, gen(Country_Region1)
recode C2M_Workplace_closing (3=2), generate(C2M_Workplace_closing1)

bysort gvkey: egen mean_atq_usd = mean(atq_usd)
xtile q_atq_usd = mean_atq_usd, nq(3)

gen qdate = quarterly(date, "YQ")
format qdate %tq
xtset url1 qdate

levelsof qdate, local(data)
capture label drop qdate
foreach d of local data {
     label define qdate `d' `"`:display %tq `d''"',  modify
}
label values qdate qdate

// gen log_atq_usd = ln(atq + 1)
gen ln_saleq = ln(saleq + 1)
gen ln_E3_Fiscal_measures = ln(E3_Fiscal_measures + 1)
destring nace_2_digit, gen (nace_2_digit1) force

replace C2M_Workplace_closing = 2 if C2M_Workplace_closing > 2
replace C6M_Stay_at_home_requirements = 2 if C6M_Stay_at_home_requirements > 2

// label var log_atq_usd "Log total assets (USD)"
// label var new_deaths_per_million "Weekly new Covid deaths per million"
// label var production_neg_sent "Website mentions covid-related production problem"
// label var demand_neg_sent "Website mentions covid-related demand problem"
// label var supply_neg_sent "Website mentions covid-related supply problem"
// label var travel_neg_sent "Website mentions covid-related travel problem"
// label var finance_neg_sent "Website mentions covid-related finance problem"
// // label var ln_E3_Fiscal_measures "Log fiscal measures (USD)"
// label var C2M_Workplace_closing "Workplace closing"
// label define C2M_Workplace_closing 0 "No measures" 1 "Recommended closing / WFH" 2 "Required closing / WFH for some sectors" 3 "Required closing / WFH for all-but-essential sectors"
// label values C2M_Workplace_closing C2M_Workplace_closing
// label var C6M_Stay_at_home_requirements "Stay at home requirements"
// label define C6M_Stay_at_home_requirements 0 "No measures" 1 "Recommend not leaving house" 2 "Require not leaving house with exceptions" 3 "Require not leaving house with minimal exceptions"
// label values C6M_Stay_at_home_requirements C6M_Stay_at_home_requirements
// // label var fiscal_measures_pct_gdp "Fiscal measures (pct. GDP)"

label var log_atq_usd "Log total assets"
label var new_deaths_per_million "Covid deaths per M"
label var covid_mention "Covid mention"
label var affected_llama "Level of affectedness"
label define affected_llama 1 "\hspace{0.25cm}Mild" 2 "\hspace{0.25cm}Moderate" 3 "\hspace{0.25cm}Severe", replace
label values affected_llama affected_llama
label values affected_gpt affected_llama
label var production_neg_sent "Production"
label var demand_neg_sent "Demand"
label var supply_neg_sent "Supply"
label var ln_E3_Fiscal_measures "Log fiscal measures"
label var C2M_Workplace_closing "Workplace closing"
label define C2M_Workplace_closing 0 "\hspace{0.25cm}No measures" 1 "\hspace{0.25cm}Recommended" 2 "\hspace{0.25cm}Required", replace
label values C2M_Workplace_closing C2M_Workplace_closing
label var C6M_Stay_at_home_requirements "Stay at home requirements"
label define C6M_Stay_at_home_requirements 0 "\hspace{0.25cm}No measures" 1 "\hspace{0.25cm}Recommended" 2 "\hspace{0.25cm}Required", replace
label values C6M_Stay_at_home_requirements C6M_Stay_at_home_requirements

//// LLM VARS
// vl create llama_affectedness_vars = (covid_mention affected_llama slightly_affected_llama moderately_affected_llama significantly_affected_llama)
// vl create llama_scale_vars = (slightly_affected_llama moderately_affected_llama significantly_affected_llama)
vl create llama_affectedness_category = (production_affected_llama demand_affected_llama supply_affected_llama)
vl create llama_affectedness_tag_vars = (hygiene_measures_llama remote_work_llama supply_chain_issues_llama closure_llama other_llama)
vl create llama_table_1_vars = (covid_mention affected_llama)

vl create gpt_affectedness_vars = (covid_mention affected_gpt)
vl create gpt_affectedness_category = (production_affected_gpt demand_affected_gpt supply_affected_gpt)
vl create gpt_affectedness_tag_vars = (hygiene_measures_gpt remote_work_gpt supply_chain_issues_gpt closure_gpt other_gpt)
vl create gpt_table_1_vars = (covid_mention affected_gpt)


vl create depvars = (log_diff_saleq_pct log_diff_stock_closing_pct)

replace covid_mention = 1 if affected_llama > 0
replace covid_mention = 1 if affected_gpt > 0
generate affected_dummy_llama = 0
replace affected_dummy_llama = 1 if affected_llama > 0

set emptycells drop
foreach depvar in $depvars {
// // baseline
// reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// reghdfe `depvar' covid_mention i.affected_llama $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
// // interacted
// reghdfe `depvar' covid_mention#i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// reghdfe `depvar' covid_mention#i.affected_llama $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
// // tags together with dummy interacted
// reghdfe `depvar' covid_mention affected_dummy_llama $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// reghdfe `depvar' covid_mention affected_dummy_llama $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
// // tags together
// reghdfe `depvar' $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// reghdfe `depvar' $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
// tags separately
foreach exp_var in $llama_affectedness_tag_vars {
// reghdfe `depvar' `exp_var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// reghdfe `depvar' `exp_var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
}
}

// both dependent in one table
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth" "Log stock returns", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

//// Robustness checks
// 2018-2021
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_2018_2021.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_2018_2021.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

// 2019-2020
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_2019_2020.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_2019_2020.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

// Manufacturing
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_manufacturing.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_manufacturing.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


// Services
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_services.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_services.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


// US
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_NA.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_NA.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

// EU
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_europe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_europe.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

// Asia
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_asia.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_asia.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


// Lagged dependent
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' L1.`depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' L1.`depvar' covid_mention i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' L1.`depvar' covid_mention i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_1_lag_dependendent.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_llama "Affected" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth (\%)" "Log stock returns (\%)", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe_with_covid_mention_1_lag_dependendent.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


// both dependent in one table ChatGPT
set emptycells drop
est clear
local i = 0
foreach depvar in $depvars {
reghdfe `depvar' covid_mention i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_on_llm_affectedness_vars_gpt_ffe_with_covid_mention.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant booktabs nomtitle replace star(* 0.05 ** 0.01 *** 0.001) addnotes("Standard errors clustered at the firm level")  refcat(1.affected_gpt "Affected (ChatGPT)" 1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap mgroups("Log sales growth" "Log stock returns", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) width(\hsize) 

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_on_llm_affectedness_vars_gpt_ffe_with_covid_mention.csv", se noomitted nobaselevels noconstant nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore




vl create llama_use_tags = (hygiene_measures_llama remote_work_llama supply_chain_issues_llama closure_llama)

set emptycells drop
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $llama_use_tags{
reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_tags_llama_3_1_manual_logits4_ffe_with_covid_mention.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_tags_llama_3_1_manual_logits4_ffe_with_covid_mention.csv", se noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}


set emptycells drop
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $llama_use_tags{
reghdfe `depvar' covid_mention `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels noomitted
estimates store policy`i'
local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_tags_llama_3_1_manual_logits4_ffe_with_covid_mention.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_tags_llama_3_1_manual_logits4_ffe_with_covid_mention.csv", se noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}


/// New version: covid mention + i.affectedness GPT
local i = 0
foreach depvar in $depvars {
est clear
reghdfe `depvar' covid_mention#i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention#i.affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' covid_mention#i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_gpt_ffe_with_covid_mention.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_gpt_ffe_with_covid_mention.csv", se noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}




/// Table 1 affectedness continuous + joint model
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $llama_table_1_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $llama_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_scale_vars L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE") // nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE"
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_llama_3_1_manual_logits4_ffe.csv", se noomitted nobaselevels noconstant order(covid_mention affected_llama $llama_scale_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}

// same for gpt vars
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $gpt_table_1_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $gpt_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $gpt_scale_vars L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $gpt_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1


* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_gpt_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention affected_gpt slightly_affected_gpt moderately_affected_gpt significantly_affected_gpt) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_vars_gpt_ffe.csv", se noomitted nobaselevels noconstant order(covid_mention affected_gpt slightly_affected_gpt moderately_affected_gpt significantly_affected_gpt) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}


/// Table 2 Sales growth on affectedness category (production, demand, supply)
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $llama_affectedness_category {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_llama_3_1_manual_logits4_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(llama_affectedness_category) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_llama_3_1_manual_logits4_ffe.csv", se noomitted nobaselevels noconstant order(llama_affectedness_category) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}

// same for gpt vars CURRENTLY ALL 0 BECAUSE OF CODING ERROR
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $gpt_affectedness_category {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_category L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_gpt_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(gpt_affectedness_category) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_gpt_ffe.csv", se noomitted nobaselevels noconstant order(gpt_affectedness_category) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}


/// table 3 llm_affectedness_tag_vars
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $llama_affectedness_tag_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_tag_vars L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $llama_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" loc1#nace_2_digit1#qdate "Country-Industry-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_llama_3_1_manual_logits4_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order($llama_affectedness_tag_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_llama_3_1_manual_logits4_ffe.csv", se noomitted nobaselevels noconstant order($llama_affectedness_tag_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}

// same for GPT vars
local i = 0
foreach depvar in $depvars {
est clear
foreach var in $gpt_affectedness_tag_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe `depvar' $gpt_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $gpt_affectedness_tag_vars L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe `depvar' $gpt_affectedness_tag_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_gpt_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order($gpt_affectedness_tag_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_gpt_ffe.csv", se noomitted nobaselevels noconstant order($gpt_affectedness_tag_vars) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore
}



/// baseline regressions
reghdfe log_diff_saleq_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

reghdfe log_diff_stock_closing_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

// baseline with different paragraph expiry
reghdfe log_diff_saleq_pct affected_w_expiry_3_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_w_expiry_6_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_w_expiry_12_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels // 12 months works best, better than baseline

reghdfe log_diff_stock_closing_pct affected_w_expiry_3_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_w_expiry_6_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_w_expiry_12_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels // 12 months works best, slightly worse than baseline

// industry-country-quarter FE
reghdfe log_diff_saleq_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.loc1#i.qdate) vce(cluster url1) baselevels 
reghdfe log_diff_saleq_pct $llama_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.loc1#i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.loc1#i.qdate) vce(cluster url1) baselevels 
reghdfe log_diff_stock_closing_pct $llama_scale_vars L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.loc1#i.qdate) vce(cluster url1) baselevels

// sales growth instead of log differences sales
reghdfe saleq_usd_gr affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels // coefficient is about half and less significant

// returns instead of log difference stock closing
reghdfe returnq affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels // coefficient is slightly higher

// baseline regressions gpt
reghdfe log_diff_saleq_pct affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

reghdfe log_diff_stock_closing_pct affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

// scale as separate categories
reghdfe log_diff_saleq_pct i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

reghdfe log_diff_stock_closing_pct i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct i.affected_llama L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct i.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

reghdfe log_diff_saleq_pct i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct i.affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

reghdfe log_diff_stock_closing_pct i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct i.affected_gpt L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct i.affected_gpt L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 

/// affectedness categories
foreach depvar in $depvars {
foreach var in $llama_affectedness_category {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
}
}
reghdfe log_diff_saleq_pct production_affected_llama demand_affected_llama supply_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct production_affected_llama demand_affected_llama supply_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

// affectedness category interacted with intensity
foreach depvar in $depvars {
foreach var in $llama_affectedness_category {
	reghdfe `depvar' c.affected_llama#`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
}
}
reghdfe log_diff_saleq_pct c.affected_llama#c.production_affected_llama c.affected_llama#c.demand_affected_llama c.affected_llama#c.supply_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct c.affected_llama#c.production_affected_llama c.affected_llama#c.demand_affected_llama c.affected_llama#c.supply_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

// Sales growth llm_affectedness_category interacted with intensity
est clear
local i = 0
foreach depvar in $depvars {
foreach var in $llm_affectedness_category {
	reghdfe `depvar' c.`var'##c.affected L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' c.`var'##c.affected L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' c.`var'##c.affected L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_interacted_intensity_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_category_interacted_intensity_ffe.csv", se noomitted nobaselevels noconstant  refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


// llm_affectedness_tag_vars interacted with intensity
est clear
local i = 0
foreach var in $llm_affectedness_tag_vars {
	reghdfe `depvar' c.`var'#c.affected L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe `depvar' c.`var'#c.affected L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe `depvar' c.`var'#c.affected L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.loc1#i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}


* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_interacted_intensity_ffe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_`depvar'_on_llm_affectedness_tag_interacted_intensity_ffe.csv", se noomitted nobaselevels noconstant refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

}

/// tags
foreach depvar in $depvars {
foreach var in $llama_affectedness_tag_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
}
}

foreach depvar in $depvars {
foreach var in $gpt_affectedness_tag_vars {
	reghdfe `depvar' `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
}
}

/// with Covid mention
reghdfe log_diff_saleq_pct covid_mention affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_saleq_pct covid_mention slightly_affected_llama moderately_affected_llama significantly_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

reghdfe log_diff_stock_closing_pct covid_mention affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct covid_mention slightly_affected_llama moderately_affected_llama significantly_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

// lead indicators: stronger effect on returns
reghdfe log_diff_saleq_pct F1.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

reghdfe log_diff_stock_closing_pct F1.affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
reghdfe log_diff_stock_closing_pct F1.slightly_affected_llama F1.moderately_affected_llama F1.significantly_affected_llama L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels


//// correlations with other measures

/// Oxford Policy Tracker
// with stringency index (country level)
correlate covid_mention StringencyIndex_Average
correlate affected_llama StringencyIndex_Average
correlate affected_gpt StringencyIndex_Average

correlate covid_mention StringencyIndex_Average if (fyearq >= 2020 & fyearq <= 2022) // low if only taking pandemic years

// with stringency index (US states, state level)
correlate covid_mention StringencyIndex_Average_state
correlate affected_llama StringencyIndex_Average_state
correlate affected_gpt StringencyIndex_Average_state

// with workplace closures
correlate covid_mention C2M_Workplace_closing
correlate affected_llama C2M_Workplace_closing
correlate affected_gpt C2M_Workplace_closing

correlate closure_llama C2M_Workplace_closing
correlate closure_gpt C2M_Workplace_closing

// remote work tag with workplace closures
correlate remote_work_llama C2M_Workplace_closing_state
correlate remote_work_gpt C2M_Workplace_closing_state

/// Epidemiological variables
// deaths per million
correlate covid_mention new_deaths_per_million
correlate affected_llama new_deaths_per_million
correlate affected_gpt new_deaths_per_million

/// Brynjolfsson wfh readiness from jobs postings (firm level, ~2000 firms left)
gen wfh_regulation = C2M_Workplace_closing_state >= 2
correlate remote_work_llama wfh_index
correlate remote_work_gpt wfh_index

// interacted with wfh regulation
gen wfh_regulation_and_index = wfh_regulation*wfh_index
correlate remote_work_llama wfh_regulation_and_index
correlate remote_work_gpt wfh_regulation_and_index

correlate covid_mention wfh_regulation_and_index

gen wfh_regulation_and_index_top_q = wfh_regulation*wfh_index_top_quartile
correlate remote_work_llama wfh_regulation_and_index_top_q

reg remote_work_gpt wfh_regulation#wfh_index_top_quartile
reg remote_work_gpt c.wfh_regulation#c.wfh_index
reg remote_work_llama c.wfh_regulation#c.wfh_index
reg remote_work_llama wfh_regulation#c.wfh_index



//// IV Regressions

/// State level Covid policies as instrument
gen wfh_regulation = C2M_Workplace_closing_state >= 2

// none of these work, huge SE
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (covid_mention = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels

ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (covid_mention = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (affected_llama = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (affected_gpt = StringencyIndex_Average_state) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels


/// software costs interacted with state level covid policies as instrument (software costs available for 1347 firms, normalized by share of revenue)
gen capsft_share_of_rev = capsft/saleq_usd_gr

// huge SE, coefficient jumping around
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (covid_mention = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (production_neg_sent = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels

ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (covid_mention = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (production_neg_sent = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (affected_llama = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (affected_gpt = c.StringencyIndex_Average_state##c.capsft_share_of_rev) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels


/// bloom industry-level remote work data (from job postings) interacted with state-level stringency 
gen inverse_bloom_remote_work_idx = 1-remote_work_pct

// cluster on industry-state level
encode(state), gen(state1)
encode(naics_3_digit), gen(naics_3_digit1)
gen industry_state_cluster_id = state1 * 100000 + naics_3_digit1 

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.StringencyIndex_Average_state#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.StringencyIndex_Average_state#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels // significant
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_llama = c.StringencyIndex_Average_state#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_llama = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_gpt = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_gpt = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels

ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (remote_work_gpt = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels
ivreghdfe log_diff_stock_closing_pct L1.log_atq_usd (remote_work_gpt = c.wfh_regulation#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster industry_state_cluster_id) baselevels


///  Brynjolfsson wfh readiness from jobs postings (firm level, ~2000 firms left) interacted with state-level stringency
gen inverse_wfh_idx = 1-wfh_index
gen inverse_wfh_idx_top_quartile = 1 - wfh_index_top_quartile

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.StringencyIndex_Average_state#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.wfh_regulation#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.StringencyIndex_Average_state#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.wfh_regulation#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_llama = c.StringencyIndex_Average_state#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_llama = c.wfh_regulation#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_gpt = c.StringencyIndex_Average_state#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_gpt = c.wfh_regulation#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster state) baselevels

// remove essential industries
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_gpt = c.StringencyIndex_Average_state#c.inverse_bloom_remote_work_idx) if (fyearq >= 2017 & fyearq <= 2022 & essential_industry==0), absorb(url1 i.qdate) vce(cluster state) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (remote_work_llama = c.wfh_regulation#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022 & essential_industry==0), absorb(url1 i.qdate) vce(cluster url1) baselevels

ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.StringencyIndex_Average_state#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022 & essential_industry==0), absorb(url1 i.qdate) vce(cluster url1) baselevels

// use indicator for covid period q1 2020 - q3 2020
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = covid_period_dummy#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022 & essential_industry==0), absorb(url1 i.qdate) vce(cluster url1) baselevels
ivreghdfe log_diff_saleq_pct L1.log_atq_usd (affected_llama = c.covid_period_dummy#c.inverse_wfh_idx) if (fyearq >= 2017 & fyearq <= 2022 & essential_industry==0), absorb(url1 i.qdate) vce(cluster url1) baselevels






////// Old indicators

vl create web_vars_reduced = (covid_mention production_neg_sent demand_neg_sent supply_neg_sent)

//// paper table: regression on covid mention, production, demand, and supply, individually and jointly with 1) just firm and quarter FE, 2) + oxford policy tracker vars, 3) + industry-quarter & country-quarter FE

// Sales growth
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1

	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model.csv", se noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore

// Stock return
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe returnq `var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe returnq `var' L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model.csv", se noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) nomtitle replace nostar plain

* Return stored estimates to their previous state
estfe . policy*, restore


//// Robustness checks: Same as above with only preferred specification for time subsamples

// Sales growth, 2018-2021
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model_2018_2021.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Sales growth, 2019-2020
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model_no_revtq_2019_2020.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Stock returns, 2018-2021
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_2018_2021.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Stock returns, 2019-2020
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2019 & fyearq <= 2020), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_2019_2020.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore


//// Robustness checks: Same as above with only preferred specification for manufacturing and services

/// Sales growth

// Manufacturing
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_rerun_07_02_24_paragraph_expiry_with_joint_model_no_revtq_manufacturing.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Services
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_rerun_07_02_24_paragraph_expiry_with_joint_model_no_revtq_services.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

/// Stock returns

// Manufacturing
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section=="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_manufacturing.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Services
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_2_digit1 >= 45  & nace_2_digit1 <= 96), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_services.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

//// Robustness checks: Same as above with only preferred specification for North America, Europe, and Asia

/// Sales growth

// NA
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model_north_america.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Europe
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model_europe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Asia
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_no_revtq_rerun_07_02_24_paragraph_expiry_with_joint_model_asia.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

/// Stock returns

// NA
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & sub_region=="Northern America"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_north_america.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Europe
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Europe"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_europe.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

// Asia
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_07_02_24_paragraph_expiry_with_joint_model_asia.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

//// Robustness checks: Same as above with one lag of dependent variable

/// Sales growth
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd L1.saleq_usd_gr if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.log_atq_usd L1.saleq_usd_gr if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_usd_gr_21_02_24_paragraph_expiry_with_joint_model_1_lag_dependendent.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

/// Stock returns
est clear
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.log_atq_usd L1.returnq if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store time_subsample`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.log_atq_usd L1.returnq if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store time_subsample`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . time_subsample*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab time_subsample* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_21_02_24_paragraph_expiry_with_joint_model_1_lag_dependendent.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . time_subsample*, restore

/////// end of code that replicates results in paper ///////









vl create web_vars = (covid_mention production_neg_sent demand_neg_sent supply_neg_sent travel_neg_sent finance_neg_sent)


// only firm and quarter FE with xtreg
foreach var in $web_vars {
	xtreg saleq_usd_gr L1.`var' L1.log_atq_usd i.qdate if (fyearq >= 2017 & fyearq <= 2022), fe robust baselevels
	estimates store simple_fe`var'
}

esttab simple_fe*, label indicate("Quarter FE = *.qdate") stats(N_g g_max N, label("No. Firms" "No. Quarters" Observations)) se r2 noomitted nobaselevels noconstant smcl compress nomtitle replace order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent)
 
esttab simple_fe* using "C:/Users/Jakob/Downloads/simple_fe.tex", label indicate("Quarter FE = *.qdate") stats(N_g g_max N, label("No. Firms" "No. Quarters" Observations)) se r2 noomitted nobaselevels noconstant tex compress nomtitle replace order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent)

// only firm and quarter FE with rehdfe
// local i = 1
// foreach var in $web_vars {
// 		reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
// 	estimates store model`i'
// 	local i=`i'+1
// // 	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
// // 	estimates store model`i'
// // 	local i=`i'+1
// }
//
// * Prepare estimates for -estout-
// 	estfe . model*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE") // loc1#qdate "County-Quarter FE")
// 	return list
//
// * Run estout/esttab
// 	esttab . model* using "C:/Users/Jakob/Downloads/regression1.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant tex compress nomtitle replace
//		
// * Return stored estimates to their previous state
// 	estfe . model*, restore



// big table: regression for each indicator and jointly with 1) just firm and quarter FE, 2) + oxford policy tracker vars, 3) + industry-quarter & country-quarter FE
local i = 0
foreach var in $web_vars {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/policy_ext_2.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) compress stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $"))

* Return stored estimates to their previous state
estfe . policy*, restore
	


//// same big table for subsamples

// subsample: exclude sector
local i = 0
foreach var in $web_vars {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & nace_section!="Manufacturing"), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/policy_non_manufacturing.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) compress stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $"))

* Return stored estimates to their previous state
estfe . policy*, restore


// subsample: world region
local i = 0
foreach var in $web_vars {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & region=="Asia"), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/asia.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) compress stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $"))
	
* Return stored estimates to their previous state
estfe . policy*, restore
	
	
// subsample: size quintiles
local i = 0
foreach var in $web_vars {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & q_atq_usd==3), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/big.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) compress stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $"))
	
* Return stored estimates to their previous state
estfe . policy*, restore


// subsample: different time horizons
local i = 0
foreach var in $web_vars {
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_usd_gr L1.`var' L1.log_atq_usd  if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2018 & fyearq <= 2021), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/2018_2021.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent L.travel_neg_sent L.finance_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) compress stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $"))
	
* Return stored estimates to their previous state
estfe . policy*, restore
	




// Industry-Quarter and REGION-quarter FE
reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.Country_Region1#i.qdate) vce(cluster url1) baselevels
reghdfe saleq_usd_gr L1.covid_mention L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
	
	
// preferred specification
xtreg saleq_gr L1.covid_mention L1.log_atq_usd i.qdate, re robust baselevels // control for size (log total assets)

// oxford policy tracker vars - index
xtreg saleq_gr L1.covid_mention L1.log_atq_usd StringencyIndex_Average new_deaths_per_million  i.qdate, re robust baselevels

// oxford policy tracker vars - categorical
xtreg saleq_gr L1.covid_mention L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements new_deaths_per_million  i.qdate, re robust baselevels

// same for 2017-2022
xtreg saleq_gr L1.production_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million  i.qdate  if (fyearq >= 2017 & fyearq <= 2022), fe robust baselevels

// same with industry-quarter FE
reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate) vce(cluster url1) baselevels

// Industry-quarter and country-quarter FE (no oxford policy vars)
reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels


// Industry-quarter, country-quarter, industry-country-quarter FE
reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate i.nace_2_digit1#i.loc1#i.qdate) vce(cluster url1) baselevels


xtreg saleq_gr L1.production_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million  i.qdate, fe robust baselevels

xtreg saleq_gr L1.covid_mention L1.log_atq_usd C1M_School_closing C2M_Workplace_closing C5M_Close_public_transport C6M_Stay_at_home_requirements PopulationVaccinated new_cases_per_million  i.qdate, re robust baselevels

['C1M_School closing', 'C2M_Workplace closing', 'C5M_Close public transport', 'C6M_Stay at home requirements', 'E1_Income support', 'E2_Debt/contract relief', 'E3_Fiscal measures', 'E4_International support', 'PopulationVaccinated', 'StringencyIndex_Average', 'StringencyIndex_Average_ForDisplay', 'GovernmentResponseIndex_Average', 'ContainmentHealthIndex_Average', 'EconomicSupportIndex', 'new_cases_per_million', 'new_deaths_per_million']


// no lag, fe
xtreg saleq_gr covid_mention i.qdate, fe robust baselevels
testparm i.qdate // test joint significance of time fixed effects

// t-1 lag works better
xtreg saleq_gr L1.covid_mention i.qdate, fe robust baselevels

xtreg saleq_gr L1.production_neg_sent i.qdate if (fyearq >= 2018 & fyearq <= 2021), fe robust baselevels


xtreg saleq_gr L1.production_neg_sent L1.log_atq_usd i.qdate if (fyearq >= 2018 & fyearq <= 2021), fe robust baselevels

// with firm, quarter, country-quarter, and industry-quarter FE
reghdfe saleq_gr L1.production_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels

reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.travel_neg_sent L1.finance_neg_sent L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels


// re
xtreg saleq_gr L1.covid_mention i.qdate, re robust baselevels
xttest0 // breusch-pagan test - H0: variance of random effects is zero

xtreg saleq_gr L1.production_neg_sent i.qdate, fe
estimates store fixed
xtreg saleq_gr L1.production_neg_sent i.qdate, re
estimates store random
hausman fixed random, sigmamore

// use FE!! Hausmann testpar, RE in appendix

// log diff as dep. variable
xtreg log_diff_saleq L1.covid_mention L1.log_atq_usd i.qdate, re robust baselevels // control for size (log total assets)

// dep. var: log sales in levels
xtreg ln_saleq L1.covid_mention L1.log_atq_usd i.qdate, re robust baselevels

// regression loop over neg_sent_vars
vl create neg_sent_vars = (production_neg_sent demand_neg_sent supply_neg_sent travel_neg_sent finance_neg_sent )´
foreach var in $neg_sent_vars {
	xtreg saleq_gr L1.`var' L1.log_atq_usd i.qdate, re robust baselevels
}


// joint model
xtreg saleq_gr L1.*_neg_sent L1.log_atq_usd i.qdate, re robust baselevels
xtreg saleq_gr L1.any_neg_sent L1.log_atq_usd i.qdate, re robust baselevels



// covid mention x size
xtreg saleq_gr L1.covid_mention L1.log_atq_usd i.qdate L1.covid_mention##L1.c.log_atq_usd, re robust baselevels

// covid mention x industry
xtreg saleq_gr L1.covid_mention i.qdate L1.covid_mention##i.nace_section1 L1.log_atq_usd, re robust baselevels 
xtreg saleq_gr L1.covid_mention i.qdate L1.covid_mention##i.wk081 L1.log_atq_usd, re robust baselevels 

// covid mention x country
xtreg saleq_gr L1.covid_mention i.qdate L1.covid_mention##i.loc1 L1.log_atq_usd, re robust baselevels 

// covid mention x time
xtreg saleq_gr L1.covid_mention L1.log_atq_usd L1.covid_mention##i.qdate, re robust baselevels

// subsample: smaller period
xtreg saleq_gr L1.covid_mention L1.log_atq_usd i.qdate  if (fyearq >= 2019 & fyearq <= 2020)

// industry-quarter FE
xtreg saleq_gr L1.covid_mention L1.log_atq_usd i.qdate i.wk081##i.qdate, fe robust baselevels

xtreg saleq_gr L1.production_neg_sent L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements new_deaths_per_million i.qdate  if (fyearq >= 2018 & fyearq <= 2021), fe robust baselevels


xtreg saleq_gr L1.production_neg_sent L1.log_atq_usd i.qdate  if (fyearq >= 2018 & fyearq <= 2021), fe robust baselevels





// exploration
// graph twoway (lfit  saleq_gr L1.covid_mention) (scatter saleq_gr L1.covid_mention)
hist saleq_gr
tab nace_section
tab wk08
xtsum saleq_gr covid_mention

hist log_diff_saleq_pct if (log_diff_saleq_pct>-10 & log_diff_saleq_pct<10)
hist log_diff_stock_closing_pct if (log_diff_stock_closing_pct>-10 & log_diff_stock_closing_pct<10)




//// regression with various lags of dependent variable

// paper table: regression for sales growth: covid mention, production, demand, and supply, and jointly with 1) just firm and quarter FE, 2) + oxford policy tracker vars, 3) + industry-quarter & country-quarter FE, NO COVID MENTION IN JOINT MODEL
local i = 0
foreach var in $web_vars_reduced {
	reghdfe saleq_gr L1.`var' L1.saleq_gr L2.saleq_gr L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_gr L1.`var' L1.saleq_gr L2.saleq_gr L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe saleq_gr L1.`var' L1.saleq_gr L2.saleq_gr L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.saleq_gr L2.saleq_gr L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.saleq_gr L2.saleq_gr L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe saleq_gr L1.production_neg_sent L1.demand_neg_sent L1.supply_neg_sent L1.saleq_gr L2.saleq_gr L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_saleq_gr_rerun_25_01_24_paragraph_expiry_l2.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . policy*, restore

// paper table: regression for stock returns: covid mention, production, demand, and supply, and jointly with 1) just firm and quarter FE, 2) + oxford policy tracker vars, 3) + industry-quarter & country-quarter FE, NO COVID MENTION IN JOINT MODEL
local i = 0
foreach var in $web_vars_reduced {
	reghdfe returnq `var' L1.returnq L2.returnq L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe returnq `var' L1.returnq L2.returnq L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
	
	reghdfe returnq `var' L1.returnq L2.returnq L1.log_atq_usd  if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels 
	estimates store policy`i'
	local i=`i'+1
}

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.returnq L2.returnq L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.returnq L2.returnq L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

reghdfe returnq production_neg_sent demand_neg_sent supply_neg_sent L1.returnq L2.returnq L1.log_atq_usd if (fyearq >= 2017 & fyearq <= 2022 & inrange(returnq, -100, 1200)), absorb(url1 i.qdate i.nace_2_digit1#i.qdate i.loc1#i.qdate) vce(cluster url1) baselevels
estimates store policy`i'
local i=`i'+1

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/full_compustat_sample_returnq_rerun_25_01_24_paragraph_expiry_l2.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(covid_mention production_neg_sent demand_neg_sent supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . policy*, restore

////

// presentation table: regression for covid mention, production, demand, and supply, and jointly with 1) just firm and quarter FE, 2) + oxford policy tracker vars, 3) + industry-quarter & country-quarter FE, NO COVID MENTION IN JOINT MODEL
local i = 0
foreach var in $web_vars_reduced {	
	reghdfe returnq L1.`var' L1.log_atq_usd i.C2M_Workplace_closing i.C6M_Stay_at_home_requirements ln_E3_Fiscal_measures new_deaths_per_million if (fyearq >= 2017 & fyearq <= 2022), absorb(url1 i.qdate) vce(cluster url1) baselevels
	estimates store policy`i'
	local i=`i'+1
}

* Prepare estimates for -estout-
estfe . policy*, labels(url1 "Firm FE" qdate "Quarter FE" nace_2_digit1#qdate "Industry-Quarter FE" loc1#qdate "Country-Quarter FE")
return list

* Run estout/esttab
esttab policy* using "C:/Users/Jakob/Downloads/regression_table_compustat_quarterly_revenue_on_indicators_policy_controls.tex", indicate(`r(indicate_fe)') se r2 label noomitted nobaselevels noconstant order(L.covid_mention L.production_neg_sent L.demand_neg_sent L.supply_neg_sent) refcat(1.C2M_Workplace_closing "Workplace closing" 1.C6M_Stay_at_home_requirements "Stay at home requirements", nolabel) tex nomtitle replace star(* 0.10 ** 0.05 *** 0.01 **** 0.001) stats(N_clust N r2, label("No. Firms" Observations "$ R^2 $")) compress wrap //longtable wrap //width(\hsize)

* Return stored estimates to their previous state
estfe . policy*, restore



