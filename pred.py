# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as pgo
import plotly.express as px

# ML-Algos
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import model_selection, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
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

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Set up function which uses an algo as input and returns the accuracy score and AUROC for the input data
def alg_fit(alg, X_train, y_train):
    
    # Model selection
    mod = alg.fit(X_train, y_train)
    
    # Prediction
    y_pred = mod.predict(X_test)
    
    # Accuracy Score
    acc_rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    print("Accuracy Score: ", acc_rmse)

    #return y_pred, acc_rmse

# ML-algos
linreg = LinearRegression()
#ridge = Ridge(alpha = a)
#lasso = Lasso(alpha = b)
xgb = XGBRegressor(n_estimators = 300,
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
alphas = np.linspace(0, 0.2, num=100)
alphas

alg_fit(linreg, X_train, y_train)
