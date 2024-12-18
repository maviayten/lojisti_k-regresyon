#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

water_quality_data = pd.read_csv(r"C:\Users\User\OneDrive - MEF Üniversitesi\Masaüstü\Lojistik_Regresyon_anaconda\waterQuality1.csv")

water_quality_data.head(20)



# In[9]:


water_quality_data.info()


# In[10]:


print(water_quality_data.describe())


# In[11]:


import pandas as pd


water_quality_data.replace('#NUM!', pd.NA, inplace=True)

for column in water_quality_data.select_dtypes(include='object').columns:
    water_quality_data[column] = pd.to_numeric(water_quality_data[column], errors='coerce')


for column in water_quality_data.columns:
    if water_quality_data[column].dtype.kind in 'biufc': 
        median_value = water_quality_data[column].median()
        water_quality_data[column].fillna(median_value, inplace=True)
    else:  
        mode_value = water_quality_data[column].mode()[0]
        water_quality_data[column].fillna(mode_value, inplace=True)

water_quality_data.info()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(24, 18))  

for i, column in enumerate(water_quality_data.select_dtypes(include='float64').columns, 1):  
 
    plt.subplot(6, 4, i) 
    sns.histplot(water_quality_data[column], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{column} Dağılımı') 
    plt.xlabel(column) 
    plt.ylabel('Frekans')  


plt.suptitle("Özniteliklerin Dağılımları", fontsize=20, y=1.02) 
plt.tight_layout()  
plt.show()


# In[15]:


import pandas as pd

def clip_outliers(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.drop('is_safe')

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

water_quality_data = clip_outliers(water_quality_data)


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(24, 18))

for i, column in enumerate(water_quality_data.select_dtypes(include='float64').columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(water_quality_data[column], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{column} Dağılımı')
    plt.xlabel(column)
    plt.ylabel('Frekans')

plt.suptitle("Özniteliklerin Dağılımları", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

water_quality_data[water_quality_data.columns[:-1]] = scaler.fit_transform(water_quality_data.drop('is_safe', axis=1))

water_quality_data.head()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(24, 18))

for i, column in enumerate(water_quality_data.select_dtypes(include='float64').columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(water_quality_data[column], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{column} Dağılımı')
    plt.xlabel(column)
    plt.ylabel('Frekans')

plt.suptitle("Özniteliklerin Dağılımları", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
correlation_matrix = water_quality_data.corr()


plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()


# In[20]:


correlation_matrix 


# In[22]:


from sklearn.model_selection import train_test_split

X = water_quality_data.drop('is_safe', axis=1) 
y = water_quality_data['is_safe'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logistic_model = LogisticRegression(random_state=42)

logistic_model.fit(X_train, y_train)

y_train_pred = logistic_model.predict(X_train)
y_test_pred = logistic_model.predict(X_test)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Eğitim Kümesi Performansı (Tam Eğitim)")
print(f"F1 Skoru: {train_f1:.4f}")
print(f"Hassasiyet: {train_precision:.4f}")
print(f"Duyarlılık: {train_recall:.4f}")
print(f"Doğruluk: {train_accuracy:.4f}")
print(100 * "*")

print("Test Kümesi Performansı")
print(f"F1 Skoru: {test_f1:.4f}")
print(f"Hassasiyet: {test_precision:.4f}")
print(f"Duyarlılık: {test_recall:.4f}")
print(f"Doğruluk: {test_accuracy:.4f}")


# In[24]:


from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

y_test_proba = logistic_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

fig = plt.figure(figsize=(20, 7))

ax1 = fig.add_subplot(1, 3, 1)
ConfusionMatrixDisplay(train_confusion_matrix).plot(ax=ax1, cmap='Blues', colorbar=False)
ax1.set_title('Eğitim Karışıklık Matrisi')

ax2 = fig.add_subplot(1, 3, 2)
ConfusionMatrixDisplay(test_confusion_matrix).plot(ax=ax2, cmap='Blues', colorbar=False)
ax2.set_title('Test Karışıklık Matrisi')

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(fpr, tpr, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
ax3.plot([0, 1], [0, 1], 'k--')
ax3.set_title('ROC Eğrisi')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend()

plt.tight_layout()
plt.show()


# In[25]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = [
    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear'], 'max_iter': [100, 200, 500]},
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 500]},
    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'max_iter': [100, 200, 500], 'l1_ratio': [0.5]}
]

log_reg = LogisticRegression()
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', verbose=1, error_score='raise')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("En İyi Parametreler:", best_params)
print("En İyi Skor:", best_score)


# In[26]:


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

best_model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=100)
best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)

print("Eğitim Veri Seti Performansı:")
print(f"Doğruluk: {train_accuracy:.4f} | F1 Skoru: {train_f1:.4f} | Hassasiyet: {train_precision:.4f} | Duyarlılık: {train_recall:.4f} | ROC AUC: {roc_auc_train:.4f}")

print("\nTest Veri Seti Performansı:")
print(f"Doğruluk: {test_accuracy:.4f} | F1 Skoru: {test_f1:.4f} | Hassasiyet: {test_precision:.4f} | Duyarlılık: {test_recall:.4f} | ROC AUC: {roc_auc_test:.4f}")


# In[27]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_test_proba = best_model.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test = auc(fpr_test, tpr_test)

train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

fig, axs = plt.subplots(2, 2, figsize=(15, 14))

ConfusionMatrixDisplay(train_conf_matrix).plot(ax=axs[0, 0], cmap='Blues', colorbar=False)
axs[0, 0].set_title('Eğitim Veri Seti - Karmaşıklık Matrisi')

ConfusionMatrixDisplay(test_conf_matrix).plot(ax=axs[0, 1], cmap='Blues', colorbar=False)
axs[0, 1].set_title('Test Veri Seti - Karmaşıklık Matrisi')

axs[1, 0].plot(fpr_train, tpr_train, label='ROC Eğrisi (AUC = %0.2f)' % roc_auc_train)
axs[1, 0].plot([0, 1], [0, 1], 'k--')
axs[1, 0].set_title('Eğitim Veri Seti - ROC Eğrisi')
axs[1, 0].set_xlabel('Yanlış Pozitif Oranı')
axs[1, 0].set_ylabel('Doğru Pozitif Oranı')
axs[1, 0].legend(loc="lower right")

axs[1, 1].plot(fpr_test, tpr_test, label='ROC Eğrisi (AUC = %0.2f)' % roc_auc_test)
axs[1, 1].plot([0, 1], [0, 1], 'k--')
axs[1, 1].set_title('Test Veri Seti - ROC Eğrisi')
axs[1, 1].set_xlabel('Yanlış Pozitif Oranı')
axs[1, 1].set_ylabel('Doğru Pozitif Oranı')
axs[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[28]:


coefficients = best_model.coef_
intercept = best_model.intercept_

print("Model Katsayıları (Coefficients):", coefficients)
print("Model Kesişim Değeri (Intercept):", intercept)


# In[ ]:




