# Project Title: Classification Model for Imbalanced Dataset
**Problem Statement:**
Car loan companies have to face huge losses due to loan defaults, which give rise to stricter policies and higher rejection rates which in turn affects the business negatively since the possibility of rejecting a potentially stable client also increases. Therefore, a financial institution wants us to create a credit risk scoring model that would help better asses the borrower’s possibility to default. This model will use several factors like previous credit history, financial stresses, and loan information of the loanee, to predict the possibility of the said loanee to default on his first installment.

**Overview**
This project aimed to develop a classification model for predicting outcomes on an imbalanced dataset. We explored multiple machine learning techniques, including Logistic Regression, Decision Trees, and Random Forests, with a focus on handling the imbalance using the Synthetic Minority Over-sampling Technique (SMOTE). The objective was to identify a model that provides the best accuracy, particularly for the minority class (Class 1).

**Methods**

**Data Preprocessing**

- **Imputation**: Missing values in numerical features were imputed using the median strategy, while categorical features were filled with a constant value ('missing').
- **Scaling and Encoding**: Numerical features were standardized using StandardScaler, and categorical features were encoded using OneHotEncoder.
- **SMOTE**: Applied SMOTE to the training data to balance the representation of the minority class.

**Model Development**

Models were trained using a pipeline that included data preprocessing, SMOTE, and the classifier itself. GridSearchCV was employed for hyperparameter tuning to optimize each model's performance.

**Results**

**1. Logistic Regression**
- Training Accuracy: 0.761
- Test Accuracy: 0.759
- Confusion Matrix (Test Data):
    - Class 0: Precision = 0.80, Recall = 0.93
    - Class 1: Precision = 0.29, Recall = 0.13

**Feature Importance:**
Coefficients of the logistic regression model were used to identify important features. Some features, such as age and income, had relatively higher coefficients indicating a stronger influence on predictions.

**Findings**:
Logistic Regression provided a balanced performance but struggled with the minority class, achieving a low recall for Class 1.
Feature importance highlighted that certain demographic features played a more significant role in the classification process.

**2. Decision Tree**
**Initial Model:**
- Training Accuracy: 0.9995
- Test Accuracy: 0.6461
- Confusion Matrix (Test Data):
    - Class 0: Precision = 0.79, Recall = 0.74
    - Class 1: Precision = 0.25, Recall = 0.31

**Findings:**
The initial decision tree model overfitted on the training data, achieving near-perfect accuracy but performed poorly on the test set, especially for Class 1.

**Tuned Model:**
- Training Accuracy: 0.7734
- Test Accuracy: 0.7678
- Confusion Matrix (Test Data):
    - Class 0: Precision = 0.79, Recall = 0.96
    - Class 1: Precision = 0.33, Recall = 0.07

**Feature Importance**:
Important features were determined based on the Gini importance from the decision tree. Features such as job_type and loan_amount were among the most influential.

**Findings**:
Despite tuning, the decision tree model showed limited improvement for Class 1. The model’s performance was skewed towards the majority class.

**3. Random Forest**
**Initial Model:**
- Training Accuracy: 1.00
- Test Accuracy: 0.7721
- Confusion Matrix (Test Data):
    - Class 0: Precision = 0.79, Recall = 0.97
    - Class 1: Precision = 0.31, Recall = 0.04
**Findings:**
- The initial Random Forest model overfitted significantly on the training data, with near-perfect precision and recall, but it struggled on the test set, particularly with the minority class.
- The model’s high complexity led to overfitting, as evidenced by the stark contrast between training and test accuracies.
**Tuned Model:**
- Training Accuracy: 0.9140
- Test Accuracy: 0.7730
- Confusion Matrix (Test Data):
    - Class 0: Precision = 0.79, Recall = 0.97
    - Class 1: Precision = 0.36, Recall = 0.06

**Feature Importance:**
Important features were determined using the average impurity reduction across the trees in the Random Forest. Features such as employment_status, credit_score, and age were identified as the most significant.

**Findings:**
The tuned Random Forest model reduced overfitting compared to the initial model, improving generalization to the test set. However, it still struggled with recall for Class 1, highlighting a persistent challenge with the minority class.
Although the tuned model showed slight improvements in precision for Class 1, recall remained low, indicating that the model was still biased towards the majority class.

**Conclusions**
After evaluating Logistic Regression, Decision Trees, and Random Forests, the models struggled to achieve satisfactory performance for the minority class (Class 1). Logistic Regression, despite its simplicity, provided the best balance between precision and recall across both classes. Therefore, it was selected as the final model because it managed the class imbalance better than more complex models like Decision Trees and Random Forests.

The feature importance analysis across all models consistently identified key demographic and financial attributes as influential in predicting outcomes, suggesting a targeted focus on these features for potential future improvements.

The final decision to use Logistic Regression was driven by its overall balanced performance, ease of interpretation, and resilience against overfitting observed in more complex models.

