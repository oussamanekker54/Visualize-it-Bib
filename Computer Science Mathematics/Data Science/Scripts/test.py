

# %% [markdown]
# ## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## 2. Load Data
df = pd.read_csv('../Data/titanic.csv')
print("Data shape:", df.shape)
df.head()

# %% [markdown]
# ## 3. Data Cleaning
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop irrelevant columns
df.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
# %% [markdown]
# ### Survival Rate by Gender
plt.figure(figsize=(8,5))
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Gender')
plt.savefig('../Outputs/Figures/survival_by_gender.png')  # Save to Outputs
plt.show()

# %% [markdown]
# ### Age Distribution of Passengers
plt.figure(figsize=(10,6))
sns.histplot(df['Age'], bins=30, kde=True, color='purple')
plt.title('Age Distribution')
plt.savefig('../Outputs/Figures/age_distribution.png')
plt.show()

# %% [markdown]
# ### Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.savefig('../Outputs/Figures/correlation_heatmap.png')
plt.show()

# %% [markdown]
# ## 5. Feature Engineering
# Create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Bin ages
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,18,35,50,100], labels=['Child','Young','Middle','Senior'])

# %% [markdown]
# ## 6. Machine Learning Prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Preprocess features
X = df.drop('Survived', axis=1)
y = df['Survived']

# Encode categorical variables
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Sex', 'Embarked', 'AgeGroup']])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# %% [markdown]
# ## 7. Model Training (Example: Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# %% [markdown]
# ## 8. Save Model
import joblib
joblib.dump(model, '../Outputs/Models/titanic_model.pkl')