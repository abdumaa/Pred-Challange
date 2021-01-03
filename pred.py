# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as pgo
import plotly.express as px

# ML-Algos
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn import model_selection, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

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

# Define ML-models and parameters to be tuned for CVgridsearch
model_params = {
    'ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
        }
    },
    'rfregressor': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'min_samples_split': [1, 2, 3],
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv = 5, return_train_score=False)
    clf.fit(X, y)

    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
scores
df_scores = pd.DataFrame(scores, columns=['model', 'best_score', 'bestparams'])
df_scores













# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Set up function which uses an algo as input and returns the accuracy score for train test split
def alg_fit(alg, X_train, y_train):
    
    # Model selection
    mod = alg.fit(X_train, y_train)
    
    # Prediction
    y_pred = mod.predict(X_test)
    
    # Accuracy Score
    acc_rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    print("Accuracy Score: ", acc_rmse)

    #return y_pred, acc_rmse

# Set up function which uses an algo as input and returns the accuracy score for cross validation
def alg_fit_cv(alg, X, y):
    
    # Cross validation
    cv = KFold(shuffle = True, random_state = 0, n_splits = 5)
    
    # Accuracy
    scores = cross_val_score(alg, X, y, cv = cv, scoring = 'neg_mean_squared_error')
    acc_rmse_cv = round(np.sqrt(scores.mean()*-1), 4)

    return acc_rmse_cv

# ML-algos
linreg = LinearRegression()
#ridge = Ridge(alpha = a)
#lasso = Lasso(alpha = b)
#xgb = XGBRegressor(n_estimators = 300,
                   max_depth = 2,
                   min_child_weight = 0,
                   gamma = 8,
                   subsample = 0.6,
                   colsample_bytree = 0.9,
                   objective = 'reg:squarederror',
                   nthread = -1,
                   scale_pos_weight = 1,
                   seed = 27,
                   learning_rate = 0.02,
                   reg_alpha = 0.006)

# Alphas for ridge and lasso
alphas = list(np.linspace(0, 2, num=101))
alphas

alg_fit(linreg, X_train, y_train)
