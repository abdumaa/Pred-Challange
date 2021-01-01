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

Show_dist('dum_floorplan')

