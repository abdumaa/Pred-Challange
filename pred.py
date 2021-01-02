# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as pgo
import plotly.express as px

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


