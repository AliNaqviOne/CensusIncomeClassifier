import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (ensure census_income.csv is in ../data/)
data = pd.read_csv('../data/census_income.csv')

# Show first few rows
print("Dataset preview:")
print(data.head())

# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Basic statistics
print("\nData description:")
print(data.describe())

# Plot distribution of the target variable
plt.figure(figsize=(6,4))
sns.countplot(x='income', data=data)
plt.title('Income Distribution')
plt.show()

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data.drop('income', axis=1)
y = data['income']

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
