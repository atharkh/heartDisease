# technical Task, Data Scientist, Sensoteq- 31/7/2023
# Athar Khodabakhsh
# Predicting Heart Disease

# Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
import pip


heart = pd.read_csv('data_01.csv', sep=',', header=0)
heart.head()

# Create features (X) and target (y) variables
y = heart.iloc[:,13]
X = heart.iloc[:,:13]

# Plot correlation matrix of heart dataset features (X)
sns.pairplot(X)
plt.tight_layout()
plt.show()

# Plot heatmap of heart dataset features (X)
cor = X.corr()
sns.heatmap(cor)
plt.tight_layout()
plt.show()


# Split the data: 80% for training, and 20% for testing.
# Train Random Forest (rf) classifier for binary classification and predicting heart disease.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make prediction on test data and form the prediction performance using accuracy, precision, and recall of the
# predicted value against ground truth.
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# The feature importance describes which features are relevant.
rf.feature_importances_
sorted_idx = rf.feature_importances_.argsort()
plt.barh(list(X.columns), rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()
plt.show()


# Plot and store the first estimator (tree) from the set of 100 estimator of Random Forest.
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = list(X.columns),
               class_names= 'target',
               filled = True);
fig.savefig('rf_individualtree.png')
