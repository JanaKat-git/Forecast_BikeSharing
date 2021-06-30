'Prediction of the bikes rented by each hour for each month(20-end) from Jan 2011 - Dec 2012'

# Import Libarys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor

#Import Data
df = pd.read_csv('TRAINDATA', parse_dates=True, index_col=0)

# Split the Data
df['h'] = df.index.hour
df['d'] = df.index.weekday
df['M'] = df.index.month
df['y'] = df.index.year
df['sunday'] = np.where(df['d'] == 6, 1, 0)

X = df[['temp','h', 'M','y','holiday','sunday','weather','windspeed','humidity']]
y = np.log(df['count'] + 1)


#Column Transformation
trans = ColumnTransformer([
   ('scale', StandardScaler(),['temp',]),
    ('ohe', OneHotEncoder(sparse=False),['h','sunday', 'M','y']),
    ('do_nothing','passthrough',['holiday', 'weather','windspeed','humidity'])
])

trans.fit(X)
X_trans = trans.transform(X)

# Train the model
m_gB = GradientBoostingRegressor(n_estimators=150, max_depth=5, min_samples_split=4)
m_gB.fit(X_trans, y) 
print('Score GradientBoosting: '+ str(round(m_gB.score(X_trans, y),2)))

#Plot Feture Importance of RandomForest
plt.figure(figsize=(10,8))
plt.bar(list(range(1,X_trans.shape[1]+1)), m_gB.feature_importances_)
plt.xlabel('No. of feature')
plt.ylabel('Gini importance')
plt.title('feature importance')

#Cross-Validation 
cross_gB = cross_validate(m_gB, X_trans, y, cv = 5, return_train_score=True)
print('Cross-Validation test: '+str(round(cross_gB['test_score'].mean(),2)) +' train: '+ str(round(cross_gB['train_score'].mean(),2)))


#MSE
def mse(ytrue, ypred):
      error = np.mean((ytrue - ypred)**2)
      return error

ypred_gB = m_gB.predict(X_trans)
mse_gB = print('MSE: ' + str(round(mse(ypred_gB, y),2)))





