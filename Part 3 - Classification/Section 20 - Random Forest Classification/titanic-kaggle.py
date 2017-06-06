# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. 
# On April 15, 1912, during her maiden voyage, the Titanic sank after colliding 
# with an iceberg, killing 1502 out of 2224 passengers and crew. 
# This sensational tragedy shocked the international community and led to better 
# safety regulations for ships.

# Importing some libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('train.csv')

# have a look at some of the stats
dataset.describe()

#fill in the missing data
dataset.Age.fillna( dataset.Age.mean() )
dataset.Fare.fillna(dataset.Fare.mean())

def plot_correlation_map( df ):
    corr = dataset.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

plot_correlation_map(dataset)

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    
# Plot distributions of Age of passangers who survived or did not survive
plot_distribution( dataset , var = 'Age' , target = 'Survived' , row = 'Sex' )
# plot distribution of fares
plot_distribution( dataset , var = 'Fare' , target = 'Survived' , row = 'Sex' )

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
# Plot survival rate by Embarked
plot_categories( dataset , cat = 'Embarked' , target = 'Survived' )
plot_categories( dataset , cat = 'Sex' , target = 'Survived' )
plot_categories( dataset , cat = 'Pclass' , target = 'Survived' )
plot_categories( dataset , cat = 'SibSp' , target = 'Survived' )
plot_categories( dataset , cat = 'Parch' , target = 'Survived' )

# split data for modeling
X = dataset.iloc[:, [2, 4, 9]].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# avoid dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#score the results
classifier.score( X_train , y_train )
classifier.score( X_test , y_test )