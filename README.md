# Exploratory Data Analysis - Customer Loans in Finance
## Description
The Customer Loans in Finance Project explores loan repayment behaviors using a dataset of loan payments. The aim is to analyse recovery percentages, identify potential revenue risks, and uncover patterns in repayment behavior based on factors like loan term, grade, and purpose.
### Objectives
- Calculate recovery percentages and project future repayments.
- Evaluate risks associated with late and charged-off loans.
- Compare repayment behaviors across various subsets (e.g., paid vs. late loans).
- Provide actionable insights for decision-making in loan management.
### What I Learned
- Data preprocessing techniques: Handling missing values, transforming skewed data, and managing outliers.
- Data visualization: Effectively presenting insights with clear and meaningful plots.
- Statistical analysis: Using chi-square tests and descriptive statistics to derive actionable insights.
## Installation Instructions
1. Clone the repository:
```bash
   git clone https://github.com/SymBains/Exploratory-Data-Analysis---Customer-Loans-in-Finance
   cd EDA
```
2. Add excluded files
- Place credentials.yaml (RDS credentials) in the root directory.
- Place loan_payments.csv (original dataset) in the root directory.
## Usage
### Data Preprocessing
Run eda.ipynb to perform data transformations, handle missing data, and output the cleaned datasets:
- transformed_loan_payments.csv: Initial transformed dataset.
- final_cleaned_dataset.csv: Fully cleaned dataset for analysis.
### Data Analysis
Run analysis.ipynb to calculate recovery metrics, analyze repayment risks, and generate visualizations.
### Outputs
- Recovery Metrics: Percentages, projected losses, and risks.
- Subgroup Insights: Loan term, grade, purpose, income, and DTI comparisons.
- Visualisations: Bar charts, boxplots, and proportional distributions.
## File Structure
```
├── credentials.yaml               # RDS database credentials (excluded by .gitignore)
├── loan_payments.csv              # Original dataset file (excluded by .gitignore)
├── transformed_loan_payments.csv  # Intermediate transformed dataset
├── final_cleaned_dataset.csv      # Fully cleaned dataset for analysis
├── eda.ipynb                      # Exploratory Data Analysis and preprocessing
├── analysis.ipynb                 # In-depth data analysis and insights generation
├── requirements.txt               # Required Python libraries
├── .gitignore                     # Excludes sensitive and large files
└── README.md                      # Project documentation
```
## Licence

This project is licensed under the MIT License.

MIT License

Copyright (c) 2024 Sym Bains

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.