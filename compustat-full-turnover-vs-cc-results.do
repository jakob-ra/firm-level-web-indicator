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
vl create llama_affectedness_category = (production_affected_llama demand_affected_llama supply_affected_llama)
vl create llama_affectedness_tag_vars = (hygiene_measures_llama remote_work_llama supply_chain_issues_llama closure_llama other_llama)
vl create llama_table_1_vars = (covid_mention affected_llama)

vl create gpt_affectedness_vars = (covid_mention affected_gpt)
vl create gpt_affectedness_category = (production_affected_gpt demand_affected_gpt supply_affected_gpt)
vl create gpt_affectedness_tag_vars = (hygiene_measures_gpt remote_work_gpt supply_chain_issues_gpt closure_gpt other_gpt)
vl create gpt_table_1_vars = (covid_mention affected_gpt)

vl create depvars = (log_diff_saleq_pct log_diff_stock_closing_pct)

generate affected_dummy_llama = 0
replace affected_dummy_llama = 1 if affected_llama > 0

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



