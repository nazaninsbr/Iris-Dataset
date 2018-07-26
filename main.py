import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

def readData():
	df = pd.read_csv('Iris.csv')
	print(df.isnull().any())
	return df 

def showData(df):
	print(df.describe())
	df['petal_width'].plot.hist()
	plt.show()
	sns.pairplot(df, hue='species')
	plt.show()

def classification(df):
	all_inputs = df[['sepal_length','sepal_width','petal_length','petal_width']].values
	all_classes = df['species'].values
	(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)
	dtc = DecisionTreeClassifier()
	dtc.fit(train_inputs, train_classes)
	print(dtc.score(test_inputs, test_classes))

if __name__ == '__main__':
	df = readData()
	showData(df)
	classification(df)