## Code to implement the decision tree  ##

# Implemented by Stephen Marsland 
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.


import numpy as np
import random
import pandas as pd

class dtree:
	""" A basic Decision Tree"""
	
	def __init__(self):
		""" Constructor """

	def prepare_train_validation_sets(self, df, bootstrap_size=0.8):
		# Create a bootstrap sample of the dataset.
		bootstrap_sample = df.sample(frac=bootstrap_size, replace=True)
		# Create the training set.
		training_set = bootstrap_sample.copy()
		feature = training_set.columns.tolist()
		# Create the validation set.
		validation_set = df.drop(bootstrap_sample.index)
		return training_set, validation_set, feature

	def data_split(self, data):
		self.classes = []
		for d in range(len(data)):
			self.classes.append(data[d][-1])
			data[d] = data[d][:-1]

		return data, self.classes

	def read_data(self,filename):
		fid = pd.read_csv(filename, index_col=False)
		Training_set, validation_set, feature = self.prepare_train_validation_sets(fid)

		Training_set.fillna("None", inplace=True)
		validation_set.fillna("None", inplace=True)

		print(Training_set)
		print()
		print(validation_set)
		print()

		Training_set = Training_set.values.tolist()
		validation_set = validation_set.values.tolist()

		train_party, train_classes = self.data_split(Training_set)
		val_party, val_classes = self.data_split(validation_set)

		return feature, train_party, train_classes, val_party, val_classes


	def classify(self,tree,datapoint):

		if type(tree) == type("string"):
			# Have reached a leaf
			return tree
		else:
			a = list(tree.keys())[0]
			for i in range(len(self.featureNames)):
				if self.featureNames[i]==a:
					break
			
			try:
				t = tree[a][datapoint[i]]
				return self.classify(t,datapoint)
			except:
				return None

	def classifyAll(self,tree,data):
		results = []
		for i in range(len(data)):
			results.append(self.classify(tree,data[i]))
		return results

	def make_tree(self,data,classes,featureNames,maxlevel=-1,level=0,forest=0):
		""" The main function, which recursively constructs the tree"""

		nData = len(data)
		nFeatures = len(data[0])

		try:
			self.featureNames
		except:
			self.featureNames = featureNames

		# List the possible classes
		newClasses = []
		for aclass in classes:
			if newClasses.count(aclass)==0:
				newClasses.append(aclass)

		# Compute the default class (and total entropy)
		frequency = np.zeros(len(newClasses))


		gini_index = 0
		index = 0
		for aclass in newClasses:
			frequency[index] = classes.count(aclass)
			gini_index += self.calc_gini(float(frequency[index])/nData)

			index += 1
		gini_index = 1 - gini_index

		default = classes[np.argmax(frequency)]

		if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
			# Have reached an empty branch
			return default
		elif classes.count(classes[0]) == nData:
			# Only 1 class remains
			return classes[0]
		else:

			# Choose which feature is best
			gain = np.zeros(nFeatures)
			featureSet = range(nFeatures)
			if forest != 0:
				np.random.shuffle(featureSet)
				featureSet = featureSet[0:forest]

			for feature in featureSet:
				g = self.calc_info_gain(data,classes,feature)
				gain[feature] = gini_index - g

			bestFeature = np.argmax(gain)
			tree = {featureNames[bestFeature]:{}}

			# List the values that bestFeature can take
			values = []
			for datapoint in data:
				if datapoint[feature] not in values:
					values.append(datapoint[bestFeature])

			for value in values:
				# Find the datapoints with each feature value
				newData = []
				newClasses = []
				index = 0
				for datapoint in data:
					if datapoint[bestFeature]==value:
						if bestFeature==0:
							newdatapoint = datapoint[1:]
							newNames = featureNames[1:]
						elif bestFeature==nFeatures:
							newdatapoint = datapoint[:-1]
							newNames = featureNames[:-1]
						else:
							newdatapoint = datapoint[:bestFeature]
							newdatapoint.extend(datapoint[bestFeature+1:])
							newNames = featureNames[:bestFeature]
							newNames.extend(featureNames[bestFeature+1:])
						newData.append(newdatapoint)
						newClasses.append(classes[index])
					index += 1

				# Now recurse to the next level	
				subtree = self.make_tree(newData,newClasses,newNames,maxlevel,level+1,forest)

				# And on returning, add the subtree on to the tree
				tree[featureNames[bestFeature]][value] = subtree

			return tree

	def printTree(self,tree,name):
		if type(tree) == dict:
			print(name, list(tree.keys())[0])
			for item in list(tree.values())[0].keys():
				print(name, item)
				self.printTree(list(tree.values())[0][item], name + "\t")
		else:
			print(name, "\t->\t", tree)

	def calc_gini(self,p):
		if p!=0:
			return p*p
		else:
			return 0

	def calc_info_gain(self,data,classes,feature):

		# Calculates the information gain based on entropy impurity
		gain = 0
		nData = len(data)

		# List the values that feature can take

		values = []
		for datapoint in data:
			if datapoint[feature] not in values:
				values.append(datapoint[feature])

		featureCounts = np.zeros(len(values))
		entropy = np.zeros(len(values))
		valueIndex = 0
		# Find where those values appear in data[feature] and the corresponding class
		for value in values:
			dataIndex = 0
			newClasses = []
			for datapoint in data:
				if datapoint[feature]==value:
					featureCounts[valueIndex]+=1
					newClasses.append(classes[dataIndex])
				dataIndex += 1

			# Get the values in newClasses
			classValues = []
			for aclass in newClasses:
				if classValues.count(aclass)==0:
					classValues.append(aclass)

			classCounts = np.zeros(len(classValues))
			classIndex = 0
			for classValue in classValues:
				for aclass in newClasses:
					if aclass == classValue:
						classCounts[classIndex]+=1 
				classIndex += 1
			
			for classIndex in range(len(classValues)):
				temp = self.calc_gini(float(classCounts[classIndex])/np.sum(classCounts))
				temp = 1 - temp
				entropy[valueIndex] += temp

			# Computes the entropy gain
			gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
			valueIndex += 1
		return gain

			
