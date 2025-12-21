"""
ML Project Pipeline: Loan Default Analysis
Converted from Jupyter Notebook to Python script
"""

# ==========================================================
# 1. Imports & Configuration
# ==========================================================
!pip install pandas
!pip install seaborn



!pip install scikit-learn


import warnings
warnings.filterwarnings('ignore')

# -- pandas and numpy --
import numpy as np
import pandas as pd

# -- plotting --
import matplotlib.pyplot as plt
import seaborn as sns

# -- sklearn stuff --

# -- sklearn modules
from sklearn.model_selection import train_test_split   #- partition train/test split
from sklearn.ensemble import RandomForestClassifier    #- random forest classifier
from sklearn.ensemble import GradientBoostingClassifier #- gradient boosting classifier
from sklearn.tree import DecisionTreeClassifier         #- decision tree classifier

# -- we need these to make our pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV # NOTE...
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# -- we need these to evaluate our models
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# -- need this to render charts in notebook --
%matplotlib inline

loan = pd.read_csv('car_loan.csv')
loan.head()

loan.columns = ( loan.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('-', '_')
    .str.replace('.', '_')
    .str.replace('(', '')
    .str.replace(')', '')
    .str.replace('?', '')
    .str.replace('\'', '') # notice the backslash \ this is an escape character
)
print(loan.columns)

# Get rid of IDs
# Identify columns containing '_id'
id_columns = [col for col in loan.columns if '_id' in col]
print(f"Columns containing '_id': {id_columns}")

# Create a list of columns to keep (excluding those with '_id')
columns_to_keep = [col for col in loan.columns if '_id' not in col]
print(f"Columns to keep: {columns_to_keep}")

# Create a new DataFrame with only the columns to keep
loan = loan[columns_to_keep]
loan.head()

loan['avg_acct_age'] = loan['average_acct_age'].str.extract(r'(\d+)').astype(float) + loan['average_acct_age'].str.extract(r'(\d+)m').fillna(0).astype(float) / 12
loan['avg_acct_age'].describe()


# ==========================================================
# 2. Data Loading
# ==========================================================
loan['credit_hist_leng'] = loan['credit_history_length'].str.extract(r'(\d+)').astype(float) + loan['credit_history_length'].str.extract(r'(\d+)m').fillna(0).astype(float) / 12
# loan['credit_hist_leng'].describe()
loan['credit_hist_leng'].head()


# Function to convert date strings based on format
def convert_dates(date_series):
    # Convert dates assuming format DD/MM/YYYY
    converted_dates_1 = pd.to_datetime(date_series, format='%d/%m/%Y', errors='coerce')

    # Convert dates assuming format DD-MM-YY (European style)
    converted_dates_2 = pd.to_datetime(date_series, format='%d-%m-%y', errors='coerce')

    # Combine both conversions, filling NaTs from the first with values from the second
    final_dates = converted_dates_1.fillna(converted_dates_2)

    return final_dates


# Apply the function to the 'date_of_birth' column
loan['date_of_birth'] = convert_dates(loan['date_of_birth'])


loan['birth_year'] = loan['date_of_birth'].dt.year

# Display the updated DataFrame
loan.head()

loan['customer_age'] = 2024 - loan['birth_year']
# deal with age < 0
loan['customer_age'] = np.where(loan['customer_age'] < 0, loan['customer_age'].median(), loan['customer_age'])
loan['customer_age'].describe()

len(loan.columns)

loan.isna().sum()

loan['employment_type'].fillna('Unknown', inplace=True)
loan['employment_type'].value_counts()

loan.drop(columns=['average_acct_age', 'credit_history_length', 'date_of_birth'], inplace=True)


# ==========================================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of important features to check skewness
important_features = [
    'disbursed_amount', 'asset_cost', 'ltv', 
    'pri_current_balance', 'pri_sanctioned_amount', 'pri_disbursed_amount', 
    'sec_current_balance', 'sec_sanctioned_amount', 'sec_disbursed_amount', 
    'primary_instal_amt', 'sec_instal_amt', 
    'pri_no_of_accts', 'pri_active_accts', 'pri_overdue_accts', 
    'sec_no_of_accts', 'sec_active_accts', 'sec_overdue_accts', 
    'no_of_inquiries', 'perform_cns_score', 'customer_age'
]

# Calculate skewness for each important feature
skewness = loan[important_features].skew()

# Print skewness values
print("Skewness of important features:\n", skewness)

# Optionally, you can plot histograms for each feature to visualize skewness
for feature in important_features:
    plt.figure(figsize=(6, 4))
    plt.hist(loan[feature].dropna(), bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of {feature} (Skewness: {skewness[feature]:.2f})')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and you already have the skewness calculated
skewed_features = ['disbursed_amount', 'asset_cost', 'pri_current_balance', 
                   'pri_sanctioned_amount', 'pri_disbursed_amount', 
                   'sec_current_balance', 'sec_sanctioned_amount', 
                   'sec_disbursed_amount', 'primary_instal_amt', 
                   'sec_instal_amt', 'pri_no_of_accts', 'pri_active_accts', 
                   'pri_overdue_accts', 'sec_no_of_accts', 'sec_active_accts', 
                   'sec_overdue_accts', 'no_of_inquiries']

# Apply log transformation to reduce positive skewness
for feature in skewed_features:
    # Since some values might be zero, we add 1 to avoid log(0)
    loan[feature] = np.log1p(loan[feature])

# Verify the effect of transformation
skewness_after_transformation = loan[skewed_features].apply(lambda x: x.skew())
print(skewness_after_transformation)

# Plot histograms to visualize the effect of transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(loan[feature], kde=True)
    plt.title(f'{feature} after Log Transformation')
    plt.show()


loan.columns

loan['id_verification_score'] = loan['mobileno_avl_flag'] + loan['aadhar_flag'] + loan['pan_flag'] + loan['voterid_flag'] + loan['driving_flag'] + loan['passport_flag']

loan['loan_burden_ratio'] = (loan['primary_instal_amt'] + loan['sec_instal_amt'] )/loan['asset_cost']

loan['new_credit_behavior'] = loan['new_accts_in_last_six_months'] + loan['no_of_inquiries']

loan['credit_stability'] = loan['credit_hist_leng'] + loan['avg_acct_age']

# ==========================================================
# 4. Data Cleaning & Preprocessing
# ==========================================================
loan['loan_default'].value_counts()

loan['loan_default'].value_counts(normalize=True).plot(kind='bar',color='lightseagreen')


loan['loan_default'].value_counts(normalize=True)

plt.figure(figsize=(8, 3))
sns.countplot(data=loan, x='loan_default', hue='loan_default')
plt.xlabel('Loan Default')
plt.title('Distribution of Loan Defaults')
plt.show()

loan.groupby('loan_default').describe().T

sns.boxplot(x='loan_default', y='customer_age', data=loan)
plt.title('Age vs Loan Default')
plt.show()

sns.boxplot(x='loan_default', y='loan_burden_ratio', data=loan)
plt.title('Loan Burden Ratio vs Loan Default')
plt.show()


# ==========================================================
# 5. Feature Engineering
# ==========================================================
sns.histplot(data=loan, x='perform_cns_score', hue='loan_default', kde=True, stat="density", common_norm=False)
plt.title('Bureau Score Distribution by Loan Default Status')
plt.show()


sns.barplot(x='loan_default', y='id_verification_score', data=loan, ci=None)
plt.title('ID Verification Score vs Loan Default')
plt.show()


sns.boxplot(x='loan_default', y='credit_hist_leng', data=loan)
plt.title('Credit Stability vs Loan Default')
plt.show()


sns.boxplot(x='loan_default', y='new_credit_behavior', data=loan)
plt.title('New Credit Behavior vs Loan Default')
plt.show()

sns.boxplot(x='loan_default', y='ltv', data=loan)
plt.title('LTV Ratio vs Loan Default')
plt.show()


sns.barplot(x='employment_type', y='loan_default', data=loan, ci=None)
plt.title('Employment Type vs Loan Default')
plt.show()


sns.boxplot(x='loan_default', y='no_of_inquiries', data=loan)
plt.title('Number of Inquiries vs Loan Default')
plt.show() 

# ==========================================================
# 6. Train / Test Split
# ==========================================================
plt.figure(figsize=(8, 5))
sns.countplot(data=loan, x='new_credit_behavior', hue='loan_default', palette='Set1')
plt.title('Loan Default by new_credit_behavior')
plt.xlabel('new_credit_behavior')
plt.ylabel('Count')
plt.legend(title='Loan Default')
plt.tight_layout()
plt.show()

# Calculate the number of unique values
num_unique_values = loan['customer_age'].nunique()
print(f"Number of unique values in 'customer_age': {num_unique_values}")

# Decide on the number of bins
num_bins = min(num_unique_values, 30)
print(f"Number of bins: {num_bins}")

# Create bins using pd.cut for non-NaN values to determine bin edges
bins = pd.cut(loan['customer_age'].dropna(), bins=num_bins, right=False, duplicates='drop')

# Get the bin edges from the resulting pd.IntervalIndex
bin_edges = bins.cat.categories

# Create a new column with binned ages using the bin edges
loan['age_bins'] = pd.cut(loan['customer_age'], bins=bin_edges, right=False, include_lowest=True)

# Ensure the age_bins column is categorical
loan['age_bins'] = loan['age_bins'].astype('category')

# Calculate the percentage distribution of loan defaults in each age bin
crosstab_result = pd.crosstab(loan['age_bins'], loan['loan_default'], normalize='index') * 100

# Convert interval index to string labels for plotting
crosstab_result.index = crosstab_result.index.map(lambda x: f'{x.left}-{x.right}')

# Plot the stacked bar chart
plt.figure(figsize=(10, 5))
plt.bar(crosstab_result.index, crosstab_result[0], label='No Default', color='lightblue', alpha=0.8)
plt.bar(crosstab_result.index, crosstab_result[1], bottom=crosstab_result[0], label='Default', color='red', alpha=0.8)
plt.title('Age Bins and Loan Defaults')
plt.xlabel('Age Bins')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Loan Default Status')
plt.tight_layout()
plt.show()

loan.drop(columns=['age_bins'], inplace=True)


def profile_numeric(col):
  # Calculate the number of unique values
  num_unique_values = loan[col].nunique()
  print(f"Number of unique values in {col}: {num_unique_values}")

  # Decide on the number of bins
  num_bins = min(num_unique_values, 30)
  print(f"Number of bins: {num_bins}")

  # Create bins using pd.cut for non-NaN values to determine bin edges
  bins = pd.cut(loan[col].dropna(), bins=num_bins, right=False, duplicates='drop')

  # Get the bin edges from the resulting pd.IntervalIndex
  bin_edges = bins.cat.categories

  # Create a new column with binned ages using the bin edges
  loan[f'{col}_bins'] = pd.cut(loan[col], bins=bin_edges, right=False, include_lowest=True)

  # Ensure the age_bins column is categorical
  loan[f'{col}_bins'] = loan[f'{col}_bins'].astype('category')

  # Calculate the percentage distribution of loan defaults in each age bin
  crosstab_result = pd.crosstab(loan[f'{col}_bins'], loan['loan_default'], normalize='index') * 100

  # Convert interval index to string labels for plotting
  crosstab_result.index = crosstab_result.index.map(lambda x: f'{x.left}-{x.right}')

  # Plot the stacked bar chart
  plt.figure(figsize=(10, 5))
  plt.bar(crosstab_result.index, crosstab_result[0], label='No Default', color='lightblue', alpha=0.8)
  plt.bar(crosstab_result.index, crosstab_result[1], bottom=crosstab_result[0], label='Default', color='red', alpha=0.8)
  plt.title(f'{col}_bins and Loan Defaults')
  plt.xlabel(f'{col}_bins')
  plt.ylabel('Percentage')
  plt.xticks(rotation=45, ha='right')
  plt.legend(title='Loan Default Status')
  plt.tight_layout()
  plt.show()
  loan.drop(columns=[f'{col}_bins'], inplace=True)

profile_numeric('avg_acct_age')

numeric_features = loan.select_dtypes(include=['int64', 'float64']).columns
numeric_features

for col in numeric_features:
  profile_numeric(col)

plt.figure(figsize=(8, 5))
sns.countplot(data=loan, x='employment_type', hue='loan_default', palette='Set1')
plt.title('Loan Default by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Count')
plt.legend(title='Loan Default')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=loan, x='loan_default', y='disbursed_amount', palette='Set1')
plt.title('Disbursed Amount by Loan Default Status')
plt.xlabel('Loan Default')
plt.ylabel('Disbursed Amount')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=loan, x='no_of_inquiries', hue='loan_default', palette='Set1')
plt.title('Loan Default by Number of Inquiries')
plt.xlabel('Number of Inquiries')
plt.ylabel('Count')
plt.legend(title='Loan Default')
plt.tight_layout()
plt.show()


# ==========================================================
# 7. Model Training
# ==========================================================
flag_vars = ['aadhar_flag', 'pan_flag', 'voterid_flag', 'driving_flag', 'passport_flag']
for flag in flag_vars:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=loan, x=flag, hue='loan_default', palette='Set1')
    plt.title(f'Loan Default by {flag.replace("_", " ").title()}')
    plt.xlabel(flag.replace("_", " ").title())
    plt.ylabel('Count')
    plt.legend(title='Loan Default')
    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = loan.select_dtypes(include=['int64', 'float64']).corr()

# Plot a heatmap to visualize the correlations
plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Remove features with high correlation (e.g., above 0.85)
threshold = 0.85
corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort").drop_duplicates()
high_corr_features = [col for col in correlation_matrix.columns if any(correlation_matrix[col] > threshold)]


# Drop highly correlated features based on the correlation matrix
loan = loan.drop(columns=[
    'sec_instal_amt',  # High correlation with sec_current_balance
    'pri_current_balance',  # High correlation with pri_disbursed_amount
    'sec_no_of_accts',  # High correlation with sec_active_accts
    'pri_active_accts',  # High correlation with pri_no_of_accts
    'sec_sanctioned_amount'  # High correlation with sec_disbursed_amount
])

# Proceed with the next steps like training a model, etc.


# Define the threshold based on the value of sec_no_of_accts
threshold = -0.008385

# Filter the DataFrame for values less than 0 and greater than or equal to the threshold
selected_features = correlation_matrix[(correlation_matrix['loan_default'] < 0) &
                                       (correlation_matrix['loan_default'] >= threshold)]['loan_default'].sort_values(ascending=False)
print(f"Selected Features Through correlation analysis are :")
selected_features

# Get the list of column names corresponding to selected features
columns_to_drop = selected_features.index.tolist()
columns_to_drop = [col for col in columns_to_drop if col not in ['sec_instal_amt', 'sec_sanctioned_amount', 'sec_no_of_accts']]
# Drop those columns from the dataset
loan.drop(columns=columns_to_drop, inplace=True)


loan.columns
print(len(loan.columns))


# Define feature matrix and target vector
X = loan.drop(columns=['uniqueid', 'loan_default'], axis=1)
y = loan['loan_default']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# ==========================================================
# 8. Model Evaluation
# ==========================================================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Define preprocessing pipeline for numerical and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Numeric transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical transformer pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply the transformations to the training set
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_preprocessed, y_train)

# After applying SMOTE, let's see the class distribution
print("\nAfter SMOTE:")
print(pd.Series(y_smote).value_counts(normalize=True))

# Plot the class distribution before and after SMOTE
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Before SMOTE
y_train.value_counts(normalize=True).plot(kind='bar', ax=axes[0], title='Class distribution before SMOTE')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Proportion')

# After SMOTE
pd.Series(y_smote).value_counts(normalize=True).plot(kind='bar', ax=axes[1], title='Class distribution after SMOTE')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')

plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Initialize the Logistic Regression model
logistic_regression_pipeline = Pipeline([
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# Train the Logistic Regression model on SMOTE-preprocessed data
logistic_regression_pipeline.fit(X_smote, y_smote)

# Split the SMOTE-processed data into train and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Predict on the training set
y_train_pred = logistic_regression_pipeline.predict(X_train_final)

# Evaluate the Logistic Regression model's performance on the training set
print("Training Data Evaluation")
print("Accuracy on training data:", accuracy_score(y_train_final, y_train_pred))
print("\nConfusion Matrix on training data:\n", confusion_matrix(y_train_final, y_train_pred))
print("\nClassification Report on training data:\n", classification_report(y_train_final, y_train_pred))

# Predict on the validation set
y_valid_pred = logistic_regression_pipeline.predict(X_valid)

# Evaluate the Logistic Regression model's performance on the validation set
print("Validation Data Evaluation")
print("Accuracy on validation data:", accuracy_score(y_valid, y_valid_pred))
print("\nConfusion Matrix on validation data:\n", confusion_matrix(y_valid, y_valid_pred))
print("\nClassification Report on validation data:\n", classification_report(y_valid, y_valid_pred))

# Preprocess the test data using the same pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Predict on the test data using the Logistic Regression model
y_test_pred = logistic_regression_pipeline.predict(X_test_preprocessed)

# Evaluate the model's performance on the test data
print("Test Data Evaluation")
print("Accuracy on test data:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix on test data:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report on test data:\n", classification_report(y_test, y_test_pred))

# Plot graphical confusion matrices for training, validation, and test sets
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Confusion matrix for training set
ConfusionMatrixDisplay.from_predictions(y_train_final, y_train_pred, ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix - Training Data')

# Confusion matrix for validation set
ConfusionMatrixDisplay.from_predictions(y_valid, y_valid_pred, ax=axes[1], cmap='Blues')
axes[1].set_title('Confusion Matrix - Validation Data')

# Confusion matrix for test set
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[2], cmap='Blues')
axes[2].set_title('Confusion Matrix - Test Data')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities for the validation and test sets
y_valid_prob = logistic_regression_pipeline.predict_proba(X_valid)[:, 1]
y_test_prob = logistic_regression_pipeline.predict_proba(X_test_preprocessed)[:, 1]

# Calculate ROC AUC for validation and test sets
roc_auc_valid = roc_auc_score(y_valid, y_valid_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

# Calculate ROC curves
fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr_valid, tpr_valid, color='blue', label=f'Validation ROC curve (area = {roc_auc_valid:.2f})')
ax.plot(fpr_test, tpr_test, color='red', label=f'Test ROC curve (area = {roc_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc='lower right')

plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Initialize the Logistic Regression pipeline
logistic_regression_pipeline = Pipeline([
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'logreg__solver': ['liblinear', 'saga'],  # Different solvers for optimization
    'logreg__penalty': ['l1', 'l2'],  # Regularization type
    'logreg__class_weight': [None, 'balanced']  # Handling class imbalance
}

# Initialize GridSearchCV with the logistic regression pipeline
grid_search_logreg = GridSearchCV(logistic_regression_pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the model using GridSearchCV on the SMOTE-preprocessed data
grid_search_logreg.fit(X_smote, y_smote)

# Print the best parameters and best score found by GridSearchCV
print("Best Score on tuned model:", grid_search_logreg.best_score_)
print("Best Parameters after model tuning:\n", grid_search_logreg.best_params_)

# Use the best estimator for final model evaluation on validation set
best_logreg_model = grid_search_logreg.best_estimator_


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Initialize the Logistic Regression pipeline
logistic_regression_pipeline = Pipeline([
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'logreg__solver': ['liblinear', 'saga'],  # Different solvers for optimization
    'logreg__penalty': ['l1', 'l2'],  # Regularization type
    'logreg__class_weight': [None, 'balanced']  # Handling class imbalance
}

# Initialize GridSearchCV with the logistic regression pipeline
grid_search_logreg = GridSearchCV(logistic_regression_pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the model using GridSearchCV on the SMOTE-preprocessed data
grid_search_logreg.fit(X_smote, y_smote)

# Print the best parameters and best score found by GridSearchCV
print("Best Score on tuned model:", grid_search_logreg.best_score_)
print("Best Parameters after model tuning:\n", grid_search_logreg.best_params_)

# Use the best estimator for final model evaluation on validation set
best_logreg_model = grid_search_logreg.best_estimator_


# Split the SMOTE-processed data into train and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Predict on the validation set
y_valid_pred = best_logreg_model.predict(X_valid)

# Evaluate the model's performance on the validation set
print("Validation Data Evaluation")
print("Accuracy on validation data:", accuracy_score(y_valid, y_valid_pred))
print("\nConfusion Matrix on validation data:\n", confusion_matrix(y_valid, y_valid_pred))
print("\nClassification Report on validation data:\n", classification_report(y_valid, y_valid_pred))

# Preprocess the test data using the same pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Predict on the test data using the best Logistic Regression model
y_test_pred = best_logreg_model.predict(X_test_preprocessed)

# Evaluate the model's performance on the test data
print("Test Data Evaluation")
print("Accuracy on test data:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix on test data:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report on test data:\n", classification_report(y_test, y_test_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Assuming the variables are defined and predictions are made:
# y_valid, y_valid_pred, y_test, y_test_pred

# Confusion matrix for validation set
conf_matrix_valid = confusion_matrix(y_valid, y_valid_pred)

# Confusion matrix for test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Plotting the confusion matrices
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Confusion matrix for validation set
ConfusionMatrixDisplay(conf_matrix_valid).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix - Validation Data')

# Confusion matrix for test set
ConfusionMatrixDisplay(conf_matrix_test).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('Confusion Matrix - Test Data')

plt.tight_layout()
plt.show()


# ==========================================================
# 9. Hyperparameter Tuning
# ==========================================================
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities for the validation and test sets using the best model
y_valid_prob = best_logreg_model.predict_proba(X_valid)[:, 1]
y_test_prob = best_logreg_model.predict_proba(X_test_preprocessed)[:, 1]

# Calculate ROC AUC for validation and test sets
roc_auc_valid = roc_auc_score(y_valid, y_valid_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

# Calculate ROC curves
fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting ROC for Validation Data
ax.plot(fpr_valid, tpr_valid, color='blue', label=f'Validation ROC curve (AUC = {roc_auc_valid:.2f})')

# Plotting ROC for Test Data
ax.plot(fpr_test, tpr_test, color='red', label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')

# Plotting the diagonal line (chance level)
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Setting limits, labels, and title
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')

# Adding the legend in the lower right
ax.legend(loc='lower right')

# Display the plot
plt.show()


from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the logistic regression model is trained
# This should already be done in your existing pipeline
logistic_model = logistic_regression_pipeline.named_steps['logreg']

# Retrieve the feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Get the coefficients from the logistic regression model
coefficients = logistic_model.coef_[0]

# Combine feature names with their corresponding coefficients
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(coefficients)  # Use absolute value of coefficients to show importance
})

# Sort the features by importance
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Filter to only show features with importance greater than a threshold if necessary
# Here we display all, but you can apply a filter if you prefer
# feature_importances = feature_importances.query('importance > 0.01')

# Plot the feature importances
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feature_importances, color="lightblue")
plt.title('Feature Importance in Logistic Regression')
plt.ylabel('Feature')
plt.xlabel('Importance (Absolute Coefficient)')
plt.xticks(rotation=45)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Split the SMOTE-processed data into train and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)

# Fit the model to the training data
rf_model.fit(X_train_final, y_train_final)

# Predict on the validation set
y_valid_pred = rf_model.predict(X_valid)

# Evaluate the model's performance on the validation set
print("Validation Data Evaluation")
print("Accuracy on validation data:", accuracy_score(y_valid, y_valid_pred))
print("\nConfusion Matrix on validation data:\n", confusion_matrix(y_valid, y_valid_pred))
print("\nClassification Report on validation data:\n", classification_report(y_valid, y_valid_pred))

# Preprocess the test data using the same pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Predict on the test data using the Random Forest model
y_test_pred = rf_model.predict(X_test_preprocessed)

# Evaluate the model's performance on the test data
print("Test Data Evaluation")
print("Accuracy on test data:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix on test data:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report on test data:\n", classification_report(y_test, y_test_pred))

# Evaluate the model's performance on the training data
y_train_pred = rf_model.predict(X_train_final)
print("Training Data Evaluation")
print("Accuracy on training data:", accuracy_score(y_train_final, y_train_pred))
print("\nConfusion Matrix on training data:\n", confusion_matrix(y_train_final, y_train_pred))
print("\nClassification Report on training data:\n", classification_report(y_train_final, y_train_pred))

# Plot graphical confusion matrices for training, validation, and test sets
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Confusion matrix for training set
ConfusionMatrixDisplay.from_predictions(y_train_final, y_train_pred, ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix - Training Data')

# Confusion matrix for validation set
ConfusionMatrixDisplay.from_predictions(y_valid, y_valid_pred, ax=axes[1], cmap='Blues')
axes[1].set_title('Confusion Matrix - Validation Data')

# Confusion matrix for test set
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[2], cmap='Blues')
axes[2].set_title('Confusion Matrix - Test Data')

plt.tight_layout()
plt.show()


# Get feature importances from the trained Random Forest model
importance = rf_model.feature_importances_

# Retrieve the feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Initialize a dictionary to store feature importances
feature_imp = {}

# Map the feature importances to their respective feature names
for i, v in enumerate(importance):
    items = feature_names[i].split('_')
    if items[0].isdigit():
        # Handle categorical features: combining the feature name with its one-hot encoded category
        fname = categorical_features[int(items[0])] + "_" + items[1]
        feature_imp[fname] = v
    else:
        # Handle numerical features
        feature_imp[feature_names[i]] = v

# Convert the dictionary to a DataFrame for easier manipulation
feature_imp = pd.DataFrame.from_dict(feature_imp, orient='index', columns=['importance'])

# Prepare and filter the DataFrame, focusing on features with importance greater than 0.01
feature_imp = (feature_imp
               .reset_index()
               .rename(columns={'index': 'feature'})
               .sort_values('importance', ascending=False)
               .query('importance > 0.01'))
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the feature importances
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feature_imp, color="lightblue")
plt.title('Feature Importance')
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.xticks(rotation=45)
plt.show()


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Select Important Features
# Get the names of the important features
important_features = feature_imp['feature'].tolist()
important_features = [fname.replace('num__', '') for fname in feature_imp['feature']]

# Filter the training and test data to keep only the important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]
# X_val_important = X_valid[important_features]

# Step 2: Retrain the Model with Important Features
# Initialize a new Random Forest model
rf_model_important = RandomForestClassifier(random_state=42)

# Train the model on the training data with important features
rf_model_important.fit(X_train_important, y_train)

# Step 3: Evaluate the Model
# Predict on the test and validation datasets
y_pred_test = rf_model_important.predict(X_test_important)
# y_pred_val = rf_model_important.predict(X_val_important)

# Calculate accuracy, confusion matrix, and classification report for test data
print("Test Data Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# Calculate accuracy, confusion matrix, and classification report for validation data
# print("\nValidation Data Metrics:")
# print("Accuracy:", accuracy_score(y_val, y_pred_val))
# print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def model_auc_roc(model, model_name, X_train, y_train, X_test, y_test):
    # 1. Predict on train and test using probability estimates
    train_predict_proba = model.predict_proba(X_train)[:, 1]
    test_predict_proba = model.predict_proba(X_test)[:, 1]

    # 2. Get FPR, TPR, and thresholds for train and test
    train_fpr, train_tpr, train_thr = roc_curve(y_train, train_predict_proba)
    test_fpr, test_tpr, test_thr = roc_curve(y_test, test_predict_proba)

    # 3. Calculate AUC for train and test
    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)

    # 4. Print performance
    print(f"--- {model_name} ---")
    print(f"Train AUC Score        : {train_auc:.6f}")
    print(f"Test AUC Score         : {test_auc:.6f}\n")

    # 5. FPR from 1 - 5%
    model_stat = pd.DataFrame({
        'fpr': test_fpr,
        'tpr': test_tpr,
        'threshold': test_thr
    }).round(decimals=2)

    # Get maximum threshold for each FPR group
    m = model_stat.loc[model_stat.groupby('fpr')['threshold'].idxmax()]

    print("--- TEST score thresholds ---")
    print(m[(m['fpr'] > 0.0) & (m['fpr'] <= 0.05)].reset_index(drop=True))
    print("\n")

    # 6. Plot ROC Curve for the entire range
    plt.figure(figsize=(10, 5))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training (area = {train_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='darkblue', lw=2, label=f'Testing (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axvline(x=0.05, color='r', linestyle='--', label='0.05 FPR Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 7. Plot ROC Curve focusing on 0 to 0.2 FPR range
    plt.figure(figsize=(10, 5))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training (area = {train_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='darkblue', lw=2, label=f'Testing (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axvline(x=0.05, color='r', linestyle='--', label='0.05 FPR Threshold')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (FPR 0 to 0.2)')
    plt.legend(loc="lower right")
    plt.show()

# Example usage of the function:
# Assume X_train_final, y_train_final, X_valid, y_valid are the processed data splits

model_auc_roc(rf_model, "Random Forest", X_train_final, y_train_final, X_valid, y_valid)


from sklearn.model_selection import GridSearchCV

# ==========================================================
# 10. Final Model & Saving Artifacts
# ==========================================================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Define preprocessing pipeline for numerical and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Numeric transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical transformer pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply SMOTE to handle class imbalance on preprocessed data
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)

# Initialize the RandomForestClassifier pipeline
random_forest_pipeline = Pipeline([
    ('rfclassifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for GridSearch
param_grid = {
    'rfclassifier__n_estimators': [100, 200, 300],
    'rfclassifier__min_samples_split': [2, 3, 5],
    'rfclassifier__max_depth': [4, 10, 30],
}

# Apply GridSearchCV with the specified parameter grid
grid_search = GridSearchCV(random_forest_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

# Evaluate the tuned model's performance
print("Best Score on tuned model:", grid_search.best_score_)
best_params = grid_search.best_params_
print("Best Parameters after model tuning:\n", best_params)

# Get the cross-validation results
cv_results_df = pd.DataFrame(grid_search.cv_results_)
print("\nResults for 5-fold cross-validation model training:\n", cv_results_df)

# Split the SMOTE-processed data into train and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)

# Use the best estimator from grid search for final model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_valid)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Plot the class distribution before and after SMOTE
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Before SMOTE
y_train.value_counts(normalize=True).plot(kind='bar', ax=axes[0], title='Class distribution before SMOTE')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Proportion')

# After SMOTE
pd.Series(y_train_smote).value_counts(normalize=True).plot(kind='bar', ax=axes[1], title='Class distribution after SMOTE')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')

plt.tight_layout()
plt.show()


# Assuming X_test and y_test are your test datasets.

# Preprocess the test data using the same pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Predict on the test data using the best model from grid search
y_test_pred = best_model.predict(X_test_preprocessed)

# Evaluate the model's performance on the test data
print("Test Data Evaluation")
print("Accuracy on test data:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix on test data:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report on test data:\n", classification_report(y_test, y_test_pred))


def model_auc_roc(model, model_name, X_train, y_train, X_test, y_test):
    # 1. Predict on train and test
    train_predict_proba = model.predict_proba(X_train)[:, 1]
    test_predict_proba = model.predict_proba(X_test)[:, 1]

    # 2. Get FPR, TPR, and thresholds for train and test
    train_fpr, train_tpr, train_thr = roc_curve(y_train, train_predict_proba)
    test_fpr, test_tpr, test_thr = roc_curve(y_test, test_predict_proba)

    # 3. Calculate AUC for train and test
    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)

    # 4. Print performance
    print(f"--- {model_name} ---")
    print(f"Train AUC Score        : {train_auc:.6f}")
    print(f"Test AUC Score         : {test_auc:.6f}\n")

    # 5. FPR from 1 - 5%
    model_stat = pd.DataFrame({
        'fpr': test_fpr,
        'tpr': test_tpr,
        'threshold': test_thr
    }).round(decimals=2)

    # Get maximum threshold for each FPR group
    m = model_stat.loc[model_stat.groupby('fpr')['threshold'].idxmax()]

    print("--- TEST score thresholds ---")
    print(m[(m['fpr'] > 0.0) & (m['fpr'] <= 0.05)].reset_index(drop=True))
    print("\n")

    # 6. Plot ROC Curve for the entire range
    plt.figure(figsize=(10, 5))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training (area = {train_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='darkblue', lw=2, label=f'Testing (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axvline(x=0.05, color='r', linestyle='--', label='0.05 FPR Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 7. Plot ROC Curve focusing on 0 to 0.2 FPR range
    plt.figure(figsize=(10, 5))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training (area = {train_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='darkblue', lw=2, label=f'Testing (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axvline(x=0.05, color='r', linestyle='--', label='0.05 FPR Threshold')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (FPR 0 to 0.2)')
    plt.legend(loc="lower right")
    plt.show()



model_auc_roc(clf, "Random Forest", X_train, y_train, X_test, y_test)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model on the SMOTE-preprocessed data
decision_tree.fit(X_smote, y_smote)

# Predict on the training data (after SMOTE)
y_train_pred = decision_tree.predict(X_smote)

# Predict on the test data
X_test_preprocessed = preprocessor.transform(X_test)
y_test_pred = decision_tree.predict(X_test_preprocessed)

# Calculate accuracy on training and test data
train_accuracy = accuracy_score(y_smote, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Calculate confusion matrices
conf_matrix_train = confusion_matrix(y_smote, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Evaluate the model's performance on the training data
print("Training Data Evaluation")
print("Accuracy on training data:", train_accuracy)
print("\nConfusion Matrix on training data:\n", conf_matrix_train)
print("\nClassification Report on training data:\n", classification_report(y_smote, y_train_pred))

# Evaluate the model's performance on the test data
print("Test Data Evaluation")
print("Accuracy on test data:", test_accuracy)
print("\nConfusion Matrix on test data:\n", conf_matrix_test)
print("\nClassification Report on test data:\n", classification_report(y_test, y_test_pred))

# Plot confusion matrices
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Confusion matrix for training data
ConfusionMatrixDisplay(conf_matrix_train, display_labels=decision_tree.classes_).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix - Training Data')

# Confusion matrix for test data
ConfusionMatrixDisplay(conf_matrix_test, display_labels=decision_tree.classes_).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('Confusion Matrix - Test Data')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities for the training and test sets
y_train_prob = decision_tree.predict_proba(X_smote)[:, 1]
y_test_prob = decision_tree.predict_proba(X_test_preprocessed)[:, 1]

# Calculate ROC AUC for training and test sets
roc_auc_train = roc_auc_score(y_smote, y_train_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

# Calculate ROC curves
fpr_train, tpr_train, _ = roc_curve(y_smote, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr_train, tpr_train, color='blue', label=f'Training ROC curve (area = {roc_auc_train:.2f})')
ax.plot(fpr_test, tpr_test, color='red', label=f'Test ROC curve (area = {roc_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc='lower right')

plt.show()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define preprocessing pipeline for numerical and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Numeric transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical transformer pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the pipeline including SMOTE and the Decision Tree Classifier
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('decisiontree', DecisionTreeClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'decisiontree__criterion': ['gini', 'entropy'],  # Criteria for splitting
    'decisiontree__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'decisiontree__min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'decisiontree__min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'decisiontree__max_features': [None, 'sqrt', 'log2']  # Number of features to consider for the best split
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy'
)

# Fit the model using GridSearchCV on the original training data (SMOTE is applied within the pipeline)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score found by GridSearchCV
print("Best Score on tuned model:", grid_search.best_score_)
print("Best Parameters after model tuning:\n", grid_search.best_params_)

# Use the best estimator for final model evaluation on training and test sets
best_decision_tree_model = grid_search.best_estimator_

# Evaluate on the test data
y_test_pred = best_decision_tree_model.predict(X_test)



from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluate the best estimator on the training data
y_train_pred = best_decision_tree_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate the best estimator on the test data
y_test_pred = best_decision_tree_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the results
print("Training Data Accuracy:", train_accuracy)
print("Test Data Accuracy:", test_accuracy)

# Confusion Matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix on Training Data:\n", conf_matrix_train)

# Confusion Matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix on Test Data:\n", conf_matrix_test)


