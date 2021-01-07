# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML-Algos
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn import model_selection, metrics, preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load in Data
df = pd.read_csv('./pred-challenge/Lab5_PredictionChallenge_training.csv')

# Understanding the data
df.head()
df.shape # 39241 x 13
df.columns 
df.dtypes # All variables are float or int
df.isnull().sum() # No missing values

# Distribution of features
def Show_dist(feat):
    if "dum" in feat:
        sns.countplot(x = feat, data = df)
        plt.show()
    else:
        sns.histplot(df[feat])
        plt.show()

Show_dist('dum_privateoffer')



# Correlation matrix using heatmap
sns.set(style = "white")
cor_matrix = df.loc[:, 'Rent':].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))

plt.figure(figsize = (15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
plt.show()



# Feature Engineering
# Split up postcode in regions (first 2 digits) and areas (last 3 digits)
v = df.postcode.astype(str)
df['region'], df['area'] = v.str[:2], v.str[2:]
df = df.drop(['postcode'], axis = 1)
df['region'] = df['region'].astype(int)
df['area'] = df['area'].astype(int)

# Data Preparation
# Define target and features
y = df['Rent']
X = df.iloc[:, 2:]

# Standardize X for Lasso and Ridge
scaler = preprocessing.StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ML Models
# Ridge regressor and parameters to be tuned for CVgridsearch using standardized X
ridge = Ridge()
params = {'alpha': [0, 1e-30, 1e-20, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20, 50, 100]}
clf = GridSearchCV(ridge, params, scoring="neg_root_mean_squared_error")
grid_search = clf.fit(X_scaled,y)
print(grid_search.best_score_*-1)
print(grid_search.best_params_)

# Lasso regressor and parameters to be tuned for CVgridsearch using standardized X
lasso = Lasso()
params = {'alpha': [0, 1e-30, 1e-20, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100]}
clf = GridSearchCV(lasso, params, scoring="neg_root_mean_squared_error")
grid_search = clf.fit(X_scaled,y)
print(grid_search.best_score_*-1)
print(grid_search.best_params_)

# RandomForestregressor and parameters to be tuned for CVgridsearch using normal X
rf = RandomForestRegressor(random_state=0)
params = {'bootstrap': [True, False],
          'n_estimators': [200, 400, 600, 800, 1000],
          'max_features': ['auto', 'sqrt'],
          'min_samples_split': [2, 5, 10],
          'max_depth': [70, 80, 90, 100, None],
          'min_samples_leaf': [1, 2, 4]}
clf = RandomizedSearchCV(rf, params, scoring="neg_root_mean_squared_error", random_state=42, n_jobs = -1, cv=3)
grid_search = clf.fit(X,y)
print(grid_search.best_score_*-1) #130.47057058639732
print(grid_search.best_params_) #{'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': False}

# XGBoostregressor and parameters to be tuned for CVgridsearch using normal X
xgb = XGBRegressor()
params = {'nthread':[4], 
          'learning_rate': [.03, 0.05, .07], 
          'max_depth': [5, 6, 7],
          'min_child_weight': [4],
          'subsample': [0.7],
          'colsample_bytree': [0.7],
          'n_estimators': [500, 600, 700]}
clf = GridSearchCV(xgb, params, scoring="neg_root_mean_squared_error", verbose=True, n_jobs = 5, cv=3)
grid_search = clf.fit(X,y)
print(grid_search.best_score_*-1) #120.10158838463475
print(grid_search.best_params_) #{'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 600, 'nthread': 4, 'silent': 1, 'subsample': 0.7}



# Use final model for final evaluations
# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

# Use final model on 5 fold cross validation
xgb_final = XGBRegressor(n_estimators=600,
                         colsample_bytree=0.7,
                         learning_rate=0.07,
                         max_depth=6,
                         min_child_weight=4,
                         nthread=4,
                         subsample=0.7)
cv = KFold(shuffle = True, random_state = 0, n_splits = 5)
scores = cross_val_score(xgb_final, X_train, y_train, cv = cv, scoring = "neg_root_mean_squared_error")
print(scores.mean()*-1) #116.02685025920496

# Test it on 1% unseen data
mod = xgb_final.fit(X_train, y_train)
y_pred = mod.predict(X_test)
acc_rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
print("Accuracy Score: ", acc_rmse) #114.8204




