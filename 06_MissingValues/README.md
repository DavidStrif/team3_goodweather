# Handling of missing values

**[Notebook](missing_values_adrian)**

The following is done in this notebook_

1. Overall evaluation of our dataset
2. Comparision of merged data set with raw data, to ensure no lines were droped
3. Continuity check of the relevant time window
4. Imputation of missing values in the raw data columns
4.1 loading relevant data (2013-07-01 -> 2019-07-30) from combined data outer with test.csv
4.2 Handling of missing Temperatur values by forward and backward filling
4.3 Handling of missing Bewoelkung values by forward and backward filling
4.4 Handling of missing Windgeschwindigkeit values by forward and backward filling
4.5 Handling of missing Niederschlag values by forward and backward filling
4.6 Handling of missing Kieler Woche values by filling empty lines with 0
4.7 Handling of missing Wettercode values by forward and backward filling
5. Creating output file combined_data_imputed.csv