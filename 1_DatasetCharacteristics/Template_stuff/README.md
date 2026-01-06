# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** [[Provide a direct link to your dataset. If the dataset is private, explain the reason and provide contact information for the dataset owner](https://github.com/DavidStrif/team3_goodweather/tree/main/1_DatasetCharacteristics/raw_data)]

### Dataset Characteristics
- **Number of Observations:** 
11783, temporal resolution: daily
- **Number of Features:** 
43

### Target Variable/Label
- **Label Name:** Umsatz
- **Label Type:** [Classification/Regression/Clustering/Other]
- **Label Description:** This label represents the sales made on a certain day.

- **Label Values:** [For classification: list of classes and their meanings. For regression: range of values. For other tasks: describe the label structure]
- **Label Distribution:** [Brief description of class balance for classification or value distribution for regression]

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**Example format:**
- **Feature 1 (Warengruppe):** Describes the product which has been sold (1 = Bread, 2 = Roles, 3 = Croissant, 4 = Pastry, 5 = cakes, 6 = Seasonal Products)
- **Feature 2 (Kielerwoche):** Tells, if the respective day is a day when Kieler Woche took place
- **Feature 3 (is_holiday):** Describes, wether the respective day was a holliday or a working day.

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
