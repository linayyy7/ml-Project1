#!/usr/bin/env python
# coding: utf-8

# # ✈️ Airline Passenger Satisfaction Prediction
# ## 📊 EDA and Model Development

# In[58]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")
print("✅ Libraries imported successfully!")


# ## 1. Get data

# In[59]:


# Load data
print("🛫 Loading dataset...")
df = pd.read_csv('Data/airline_passengers_satisfaction.csv')
print(f"Dataset shape: {df.shape}")

# Basic info
print("\n📋 Dataset Info:")
print("=" * 40)
print(df.info())

print("\n🔍 Missing Values:")
print("=" * 40)
print(df.isnull().sum())


# In[60]:


# Display first few rows
print("📊 First 5 rows:")
df.head()


# ## 2. Data Cleaning

# In[61]:


# Data cleaning
print("🧹 Cleaning data...")
df = df.copy()


# In[62]:


# Remove the first column (Unnamed: 0)
df = df.drop(columns=['Unnamed: 0'])

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Handle missing values
df['arrival_delay_in_minutes'] = df['arrival_delay_in_minutes'].fillna(0)
df = df.dropna()

print(f"✅ Cleaned dataset shape: {df.shape}")


# In[63]:


df.head().T


# In[64]:


# Convert satisfaction to binary (1 = satisfied, 0 = neutral/dissatisfied)
df.satisfaction = (df.satisfaction == 'satisfied').astype(int)


# ## 3. Split Dataset

# In[65]:


# Split the data into training, validation, and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Print the sizes of each dataset
len(df_train), len(df_val), len(df_test)


# In[66]:


# Reset the index for all split datasets to avoid index conflicts
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[67]:


# Extract target variable (y) from each dataset
y_train = df_train.satisfaction.values
y_val = df_val.satisfaction.values
y_test = df_test.satisfaction.values

# Remove the target column from feature sets
del df_train['satisfaction']
del df_val['satisfaction']
del df_test['satisfaction']


# ## 4. EDA

# In[68]:


df_full_train.dtypes


# In[69]:


# Define numerical and categorical features
print("🔧 Defining feature types...")

# Numerical features
numerical_features = [
     'age', 'flight_distance', 'inflight_wifi_service',
       'departure/arrival_time_convenient', 'ease_of_online_booking',
       'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
       'inflight_entertainment', 'on-board_service', 'leg_room_service',
       'baggage_handling', 'checkin_service', 'inflight_service',
       'cleanliness', 'departure_delay_in_minutes', 'arrival_delay_in_minutes']

# Categorical features
categorical_features = [
    'gender', 'customer_type', 'type_of_travel', 'class'
]

# Target variable
target_feature = 'satisfaction'

print("📊 Feature Summary:")
print("=" * 50)
print(f"Total features: {len(numerical_features) + len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Target: {target_feature}")


# In[70]:


# Numerical features analysis
df_full_train[numerical_features].describe()


# In[71]:


# Numerical features distributions
df_full_train[numerical_features].hist(bins=30, figsize=(16, 9), color='skyblue', alpha=0.7)
plt.suptitle('Numerical Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()


# In[72]:


# Target variable analysis
print("\n🎯 Target Variable Distribution:")
print("=" * 40)
target_counts = df_full_train['satisfaction'].value_counts()
print(target_counts)
print(f"\nPercentage:\n{(target_counts / len(df_full_train) * 100).round(1)}")

# Target distribution plot
df_full_train.satisfaction.value_counts().plot(kind='bar', title='Target Distribution', color='skyblue', alpha=0.7)
plt.xticks(rotation=0) 
plt.xlabel('0 = neutral or dissatisfied, 1 = satisfied')
plt.show()


# In[73]:


# Categorical features - shortest version
fig, axes = plt.subplots(2, 2, figsize=(16, 9))

for i, col in enumerate(categorical_features):
    ax = axes.flat[i]
    counts = df_full_train[col].value_counts()
    ax.bar(counts.index, counts.values, color='skyblue')
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xticklabels(counts.index, rotation=45)

plt.suptitle('Categorical Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()


# ### Feature Importance: Satisfaction Rate and Risk Ratio

# In[74]:


global_satisfaction = df_full_train.satisfaction.mean()
for c in categorical_features:
    print(c)
    df_group = df_full_train.groupby(c).satisfaction.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_satisfaction
    df_group['risk'] = df_group['mean'] / global_satisfaction
    display(df_group)
    print()
    print()


# ### Feature Importance: Correlation

# In[75]:


df_full_train[numerical_features].corrwith(df_full_train.satisfaction).sort_values()


# In[76]:


plt.figure(figsize=(15,15))
sns.heatmap(df_full_train[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# ## 4. One-hot Encoding

# In[88]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')
X_val = dv.transform(val_dict)
test_dict = df_test[categorical_features + numerical_features].to_dict(orient='records')
X_test = dv.transform(test_dict)


# In[78]:


# Column names after one-hot encoding
print(f"Features: {dv.get_feature_names_out()}")



# ## 5. Models

# In[79]:


# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

print(f"🔧 {len(models)} models ready for comparison:")
for i, name in enumerate(models.keys(), 1):
    print(f"{i}. {name}")


# In[94]:


# Train and evaluate models
results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'AUC': roc_auc_score(y_val, y_pred_proba),
        'model_object': model
    }

    results.append(metrics)
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   AUC: {metrics['AUC']:.4f}")

# Results table
results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n📋 Results:")
print("=" * 60)
print(results_df[['Model', 'Accuracy', 'F1', 'AUC']].round(4))


# In[83]:





# In[95]:


# Select best model
best_idx = results_df['AUC'].idxmax()
best_name = results_df.loc[best_idx, 'Model']
best_model = results_df.loc[best_idx, 'model_object']

print(f"🏆 Best Model: {best_name}")
print(f"   F1-Score: {results_df.loc[best_idx, 'F1']:.4f}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"   AUC-ROC: {results_df.loc[best_idx, 'AUC']:.4f}")


# In[96]:


# Random Forest Test Set Results
print("📊 Random Forest - Test Set")
print("=" * 40)

# Test set predictions
X_test = dv.transform(df_test[categorical_features + numerical_features].to_dict('records'))
y_pred = models['Random Forest'].predict(X_test)

# Results
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC :  {f1_score(y_test, y_pred):.3f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


# In[85]:


# Save model
import pickle

model_to_save = {
    'model': best_model,
    'vectorizer': dv,
    'features': categorical_features + numerical_features,
    'performance': results_df.loc[best_idx].to_dict()
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_to_save, f)

print(f"💾 Model saved: best_model.pkl ({best_name})")


# In[90]:


# Random Forest Test Set Results
print("📊 Random Forest - Test Set")
print("=" * 40)

# Test set predictions
y_pred = models['Random Forest'].predict(X_test)

# Results
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")

roc_auc_score
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




