#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

titanic_data = pd.read_csv("C:\\Users\\User\\OneDrive - MEF Üniversitesi\\Masaüstü\\Lojistik_Regresyon_anaconda\\Titanic-Dataset .csv")

titanic_data.sample(15)



# In[3]:


print(titanic_data.info())


# In[4]:


titanic_data.describe()


# In[5]:


titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


titanic_data.drop('Cabin', axis=1, inplace=True)

titanic_data


# In[6]:


titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(titanic_data['Embarked'], prefix='Embarked')
titanic_data = pd.concat([titanic_data, embarked_dummies], axis=1)
titanic_data.drop('Embarked', axis=1, inplace=True)

titanic_data


# In[7]:


titanic_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

titanic_data


# In[8]:


titanic_data.describe()


# In[9]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
titanic_data['Survived'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
plt.title('Hayatta Kalma Durumunun Dağılımı')
plt.xlabel('Hayatta Kalma Durumu (0 = Hayır, 1 = Evet)')
plt.ylabel('Yolcu Sayısı')
plt.xticks(rotation=0)
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='Sex', hue='Survived', palette=['salmon', 'lightblue'])
plt.title('Cinsiyet ve Hayatta Kalma Durumu')
plt.xlabel('Cinsiyet (0 = Erkek, 1 = Kadın)')
plt.ylabel('Yolcu Sayısı')
plt.legend(title='Hayatta Kalma Durumu', labels=['Hayır', 'Evet'])
plt.show()


# In[11]:


plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='Pclass', hue='Survived', palette=['salmon', 'lightblue'])
plt.title('Yolcu Sınıfı ve Hayatta Kalma Durumu')
plt.xlabel('Yolcu Sınıfı (1 = Birinci Sınıf, 2 = İkinci Sınıf, 3 = Üçüncü Sınıf)')
plt.ylabel('Yolcu Sayısı')
plt.legend(title='Hayatta Kalma Durumu', labels=['Hayır', 'Evet'])
plt.show()


# In[12]:


plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Age', hue='Survived', bins=30, kde=True, palette=['salmon', 'lightblue'], alpha=0.6)
plt.title('Yaş Dağılımı ve Hayatta Kalma Durumu')
plt.xlabel('Yaş')
plt.ylabel('Yolcu Sayısı')
plt.legend(title='Hayatta Kalma Durumu', labels=['Hayır', 'Evet'])
plt.show()


# In[13]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=titanic_data, x='Survived', y='Fare', palette=['salmon', 'lightblue'])
plt.title('Bilet Ücreti ve Hayatta Kalma Durumu')
plt.xlabel('Hayatta Kalma Durumu (0 = Hayır, 1 = Evet)')
plt.ylabel('Bilet Ücreti (Fare)')
plt.show()


# In[14]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=titanic_data, x='SibSp', hue='Survived', palette=['salmon', 'lightblue'])
plt.title('Eş/Kardeş Sayısı ve Hayatta Kalma Durumu')
plt.xlabel('Eş veya Kardeş Sayısı')
plt.ylabel('Yolcu Sayısı')
plt.legend(title='Hayatta Kalma Durumu', labels=['Hayır', 'Evet'])

plt.subplot(1, 2, 2)
sns.countplot(data=titanic_data, x='Parch', hue='Survived', palette=['salmon', 'lightblue'])
plt.title('Ebeveyn/Çocuk Sayısı ve Hayatta Kalma Durumu')
plt.xlabel('Ebeveyn veya Çocuk Sayısı')
plt.ylabel('Yolcu Sayısı')
plt.legend(title='Hayatta Kalma Durumu', labels=['Hayır', 'Evet'])

plt.tight_layout()
plt.show()


# In[15]:


corr_matrix = titanic_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Titanic Dataset')
plt.show()


# In[16]:


titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['IsAlone'] = 1
titanic_data['IsAlone'].loc[titanic_data['FamilySize'] > 0] = 0


# In[17]:


titanic_data


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=titanic_data, x='Fare')
plt.title('Bilet Ücreti (Fare) İçin Boxplot')

plt.subplot(1, 2, 2)
sns.boxplot(data=titanic_data, x='Age')
plt.title('Yaş (Age) İçin Boxplot')

plt.tight_layout()
plt.show()


# In[19]:


import numpy as np

fare_99_percentile = titanic_data['Fare'].quantile(0.99)

titanic_data['Fare'] = np.where(titanic_data['Fare'] > fare_99_percentile, fare_99_percentile, titanic_data['Fare'])

titanic_data


# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

titanic_data[['Age', 'Fare']] = scaler.fit_transform(titanic_data[['Age', 'Fare']])

titanic_data


# In[21]:


titanic_data.to_csv('Titanic_Cleaned.csv', index=False)


# In[22]:


from sklearn.model_selection import train_test_split

titanic_cleaned = pd.read_csv('Titanic_Cleaned.csv')

X = titanic_cleaned.drop('Survived', axis=1)
y = titanic_cleaned['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]




# In[24]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[25]:


roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[26]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'C': np.concatenate((np.linspace(0.001, 1, 1000, endpoint=False), np.arange(1, 101, 1))),
    'penalty': ['l1', 'l2',"elasticnet"],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500,1000, 2000]
}

logreg = LogisticRegression()
grid_search = RandomizedSearchCV(logreg, param_grid, cv=10, scoring='f1', verbose=1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("En İyi Hiperparametreler:", best_params)
print("Çapraz Doğrulama Doğruluğu:", best_score)


# In[27]:


print("En İyi Hiperparametreler:", best_params)
print("Çapraz Doğrulama Doğruluğu:", best_score)


# In[28]:


final_model = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver=best_params['solver'],
    max_iter=best_params['max_iter']
)
final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_precision = precision_score(y_test, final_predictions)
final_recall = recall_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions)

final_accuracy, final_precision, final_recall, final_f1


# In[29]:


coefficients = final_model.coef_
intercept = final_model.intercept_


print("Model Katsayıları (Coefficients):", coefficients)
print("Model Kesişim Değeri (Intercept):", intercept)


# In[ ]:




