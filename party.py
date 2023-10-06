## Code to run the decision tree on the Party dataset ##

# Implemented by Stephen Marsland 
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.


import dtree

predicted = []
train_predicted = []
tree = dtree.dtree()

features, train_party, train_classes, val_party, val_classes = tree.read_data(r'C:\Users\epava\ML\HW1\Part-4\party.data')

t = tree.make_tree(train_party, train_classes, features)

#print(train_party)
#print(train_classes)
#print(features)

train_predicted = tree.classifyAll(t, train_party)
val_predicted = tree.classifyAll(t, val_party)


def error_rate(classes, predicted):
    total_error = 0
    for i in range(len(classes)):
        if predicted[i] != classes[i]:
            total_error += 1

    Error_Rate = (total_error / len(classes))
    return Error_Rate


tree.printTree(t,"")

print(10*"*" + "Train Error Rate" + 10*"*")
print("Predicted Classes")
print(train_predicted)

print("True Classes")
print(train_classes)

print("Error Rate")
print(error_rate(train_classes, train_predicted))


print(10*"*" + "Test Error Rate" + 10*"*")
print("Predicted Classes")
print(val_predicted)

print("True Classes")
print(val_classes)

print("Error Rate")
print(error_rate(val_classes, val_predicted))
