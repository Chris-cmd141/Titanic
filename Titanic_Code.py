## LIBRARIES ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.metrics import accuracy_score  # Import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, svm
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from scipy import stats


## LOADING DATA SET ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import numpy as np
# Load your dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_data = pd.read_csv('gender_submission.csv')


## OVERVIEW OF THE DATASET///////////////////////////////////////////////////////////////////////////////
train_data.info() ## Based on the output I've decided to drop the cabin column (too many null values), ticket (due to the inconsistent name format) and name which is not relevant for us to predict the survival rate.
test_data.info()
train_data



##  DATA CLEANING ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## insert the survived column into the test data
test_data.insert(1, 'Survived', gender_data['Survived']) ## the values for the "Survived" column are taken from "gender_data" 

X_train = train_data.drop(['Survived','Name', 'Ticket', 'Cabin'], axis = 1) ## removes the "Survived" column from the training data, the model will be trained based on the other independent variable. We remove "Survived" column because this is what we're trying to predict. We're training the model to predict this column.
X_train ## Based on the output I've decided to drop the cabin column (too many null values), ticket (due to the inconsistent name format) and name which is not relevant for us to predict the survival rate.
Y_train = train_data['Survived'] ## I'm telling Python to create a variable Y_train which will contain a column "Survived" with the values from train_data (train.csv)
## so basically with the last 2 codes we've told Python to take the Survived column out from X, because it's a dependent variable  and to put it in Y where it will become the dependent variable used to train the mode later on.
X_train
Y_train

## same as above but appLied to testing lot
X_test = test_data.drop(['Survived','Name', 'Ticket', 'Cabin'], axis = 1)
X_test
Y_test = test_data['Survived']
Y_test

## DETECING DUPLICATES
duplicates_X_train = X_train[X_train.duplicated()]
duplicates_X_train

duplicates_X_test = X_test[X_test.duplicated()]
duplicates_X_test


## DETECTING NULL VALUES
null_values_X_train = X_train.isnull()
null_values_X_train

null_values_X_test = X_test.isnull()
null_values_X_test

missing_count_X_train = X_train.isnull().sum()
print(missing_count_X_train)

missing_count_X_test = X_test.isnull().sum()
print(missing_count_X_test)

## I NEED TO INPUT MISSING VALUES WITH THE MEAN BUT EMBARKED IS NOT A INTEGER, IS AN OBJECT, SO I NEED TO MAP IT TO CHANGE IT TO INTEGER SO I CAN CALCULATE THE MEDIAN
X_train['Embarked'] = X_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X_test['Embarked'] = X_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

## FILLING THE MISSING VALUES FOR "EMBARKED" COLUMN
median_embarked = X_train['Embarked'].median()
X_train['Embarked'].fillna(median_embarked, inplace=True)
X_test['Embarked'].fillna(median_embarked, inplace=True)


## FILLING THE MISSING VALUES FOR "AGE" COLUMN
median_age = X_train['Age'].median()
X_train['Age'].fillna(median_age, inplace=True)
X_test['Age'].fillna(median_age, inplace=True)

## FILLING THE MISSING VALUES FOR "Fare" COLUMN
# Calculate the median Fare value FIRST
median_fare = X_train['Fare'].median()

X_train['Fare'].fillna(median_fare, inplace=True)
X_test['Fare'].fillna(median_fare, inplace=True)

## THERE ALSO TWO 0 VALUES ON FARE WHICH WE ARE GOING TO REPLACE WITH THE MEDIAN

# I NEED TO REPLACE THE 0 VALUES IN THE FARE COLUMN WITH THE MEDIAN. WE CONSIDER 0 VALUE FARE TO BE
X_train['Fare'] = X_train['Fare'].replace(0, median_fare)
X_test['Fare'] = X_test['Fare'].replace(0, median_fare)


# ONE HOT ENCODING FOR THE SEX COLUMN - WE NEED TO CHANGE THE OBSERVATION MALE AND FEMALE IN INTEGER VALUES SO WE CAN USE IN A HEATMAP AND IN THE PREDICTION MODELS
X_train = pd.get_dummies(X_train, columns=["Sex"], prefix= "Sex")
X_test = pd.get_dummies(X_test, columns=["Sex"], prefix= "Sex")

X_train.info()
X_test.info()

X_train


## OUTLIERS //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



import seaborn as sns
import matplotlib.pyplot as plt

# Combine X_train and X_test for a comprehensive view of outliers
combined_data = pd.concat([X_train, X_test])

# Create a boxplot for each column
plt.figure(figsize=(16, 8))
sns.boxplot(data=combined_data)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title("Boxplot of Columns to Identify Outliers")
plt.show()


## REMOVE OUTLIERS

from scipy import stats

# Define the thresholds for 'Age' and 'Fare'
age_min_threshold = 1
age_max_threshold = 100
fare_min_threshold = 1
fare_max_threshold = 400

# Identify and remove outliers based on the thresholds X_train
X_train_no_outliers = X_train[
    (X_train['Age'] >= age_min_threshold) & (X_train['Age'] <= age_max_threshold) &
    (X_train['Fare'] >= fare_min_threshold) & (X_train['Fare'] <= fare_max_threshold)
]

# Identify and remove outliers based on the thresholds X_test
X_test_no_outliers = X_test[
    (X_test['Age'] >= age_min_threshold) & (X_test['Age'] <= age_max_threshold) &
    (X_test['Fare'] >= fare_min_threshold) & (X_test['Fare'] <= fare_max_threshold)
]


# Check the impact for X_train
print("Original X_train shape:", X_train.shape)
print("X_train shape after removing outliers:", X_train_no_outliers.shape)

# Check the impact for X_test
print("Original X_test shape:", X_test.shape)
print("X_test shape after removing outliers:", X_test_no_outliers.shape)



## BOXPLOT WITH OUTLIERS REMOVED
import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot boxplots for 'Age' and 'Fare' in X_train after removing outliers
sns.boxplot(y=X_train_no_outliers['Age'], ax=axes[0])
sns.boxplot(y=X_train_no_outliers['Fare'], ax=axes[1])

# Set titles and labels for the subplots
axes[0].set_title('Age Boxplot (Outliers Removed)')
axes[1].set_title('Fare Boxplot (Outliers Removed)')
axes[0].set_ylabel('Age')
axes[1].set_ylabel('Fare')

# Show the boxplots
plt.show()





## VISUALISATIONS ///////////////////////////////////////////////////////////////////



# CORELATION MATRIX FOR X_TRAIN after DATA CLEANING WAS DONE - IT DOES NOT INCLUDE THE Y DEPENDENT VARIABLE

# Load dataset
train_data = pd.read_csv('train.csv')
# Same data cleaning steps
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
train_data['Embarked'].fillna(train_data['Embarked'].mean(), inplace=True)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Fare'] = train_data['Fare'].replace(0, train_data['Fare'].median())
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
train_data = pd.get_dummies(train_data, columns=["Sex"], prefix="Sex")
# Drop the columns you don't need
train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# Create a correlation matrix for the cleaned dataset
corr_matrix = train_data.corr()
# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Check the correlation values between variables
correlation_AB = corr_matrix.loc['Survived', 'Pclass']
print(f'Correlation between Survived and Pclass: {correlation_AB:.2f}')

# Check the correlation values between variables
correlation_AB = corr_matrix.loc['Fare', 'Pclass']
print(f'Correlation between Fare and Pclass: {correlation_AB:.2f}')

'''
# Convert the "Survived" column to a string type
train_data["Survived"] = train_data["Survived"].astype(str)

# Create a count plot
sns.countplot(x="Sex", hue="Survived", data=train_data, palette="Set1")

# Customize the plot
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])

# Show the plot
plt.show()


# Convert the "Survived" column to a string type
train_data["Survived"] = train_data["Survived"].astype(str)
# Create a count plot
sns.countplot(x="Sex", hue="Survived", train_data=train_data, palette="Set1")
# Customize the plot
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])

# Show the plot
plt.show()
'''


## LOGISTIC REGRESSION//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#  Logistic Regression model
logistic_model = LogisticRegression()

# Fit the model on your training data
logistic_model.fit(X_train, Y_train)

# Make predictions using the test data
Y_pred = logistic_model.predict(X_test)
print(Y_pred)



## PROBABILITIES
# Calculate the predicted probabilities of survival
probabilities = logistic_model.predict_proba(X_test)

# The second column of 'probabilities' contains the probability of survival
probability_of_survival = probabilities[:, 1]

# Print the predicted probabilities
print("Predicted Probabilities of Survival:")
print(probability_of_survival)

df_probabilities = pd.DataFrame({'Probability of Survival': probability_of_survival})
df_probabilities['Probability of Survival'] = df_probabilities['Probability of Survival'] * 100
print(df_probabilities)
## PROBABILITIES END


# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)

## create a dataframe that shows the predicted and actual value
df_logistic_compare = pd.DataFrame({'Actual':Y_test, 'Predicted': Y_pred})
df_logistic_compare

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion}")

# Save the DataFrame to a CSV file
df_logistic_compare.to_csv('logistic_compare.csv', index=False)


## KNN /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# KNN 
knn = KNeighborsClassifier(n_neighbors=5)  # You can choose an appropriate value for 'n_neighbors'

# Fit the model to your training data
knn.fit(X_train, Y_train)

# Make predictions on the test data
predictions_knn = knn.predict(X_test)

# Evaluate the model's performance
accuracy = knn.score(X_test, Y_test)

print('Accuracy KNN:', metrics.accuracy_score(Y_test, predictions_knn))



## DECISION TREE /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## decision tree
## creating a decision tree with criteria gini with no max depth
decision_tree_gini = DecisionTreeClassifier()

## creating a decision tree with criteria Entropy with no max depth
decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')

## max depth of 3
decision_tree_depth = DecisionTreeClassifier(max_depth=3)

## fit our model
decision_tree_gini.fit(X_train, Y_train)
decision_tree_entropy.fit(X_train, Y_train)
decision_tree_depth.fit(X_train, Y_train)

## make predictions
y_pred_gini = decision_tree_gini.predict(X_test)
y_pred_entropy = decision_tree_entropy.predict(X_test)
y_pred_depth = decision_tree_depth.predict(X_test)

## model evaluation
print('Accuracy(Gini):', metrics.accuracy_score(Y_test, y_pred_gini))
print('Accuracy(Entropy):', metrics.accuracy_score(Y_test, y_pred_entropy))
print('Accuracy(Max Depth):', metrics.accuracy_score(Y_test, y_pred_depth))


## SUPPORT VECTOR MACHINE /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

clflinear = svm.SVC(kernel = 'linear')
clfsigmoid = svm.SVC(kernel = 'sigmoid')
clfrbf = svm.SVC(kernel = 'rbf')

## fir the model
clflinear.fit(X_train, Y_train)
clfsigmoid.fit(X_train, Y_train)
clfrbf.fit(X_train, Y_train)

## make prediction

y_pred_linear = clflinear.predict(X_test)
y_pred_sigmoid = clfsigmoid.predict(X_test)
y_pred_rbf = clfrbf.predict(X_test)


## model evaluation
print('Accuracy(Linear Kernel):', metrics.accuracy_score(Y_test, y_pred_linear))
print('Accuracy(Sigmoid Kernel):', metrics.accuracy_score(Y_test, y_pred_sigmoid))
print('Accuracy(RBF Kernel):', metrics.accuracy_score(Y_test, y_pred_rbf))

print(classification_report(Y_test, y_pred_sigmoid))



## PREDICT IF A PERSON SURVIVED OR NOT BASED ON CHOSEN INDEPENDENT VARIABLES

X_train

## I create a new DF with the variables that I want 
new_data = pd.DataFrame({
    'PassengerId': [0],
    'Pclass': [1],
    'Age': [30],
    'SibSp': [0],  # I will choose the values that I want to predict for a specific person the survival score
    'Parch': [0],
    'Fare': [200],
    'Embarked': [0],
    'Sex_female': [1], ## If it's female, I will write 1. If it's not female it will be 0. 1 for True, 0 for False.
    'Sex_male': [0],
   
})


## Then I choose the model that I want to make the prediction for me
new_prediction = logistic_model.predict(new_data)
print(f"The survival predicted: {new_prediction}")



## PROBABILITIES FOR ONE FICTIONAL INDIVIDUAL
# Calculate the predicted probabilities of survival
probabilities_fictional = logistic_model.predict_proba(new_data)

# The second column of 'probabilities' contains the probability of survival
probability_of_survival_fictional = probabilities_fictional[:, 1]

# Print the predicted probabilities
print("Predicted Probabilities of Survival:")
print(probability_of_survival_fictional)

df_probabilities_fictional = pd.DataFrame({'Probability of Survival': probability_of_survival_fictional})
df_probabilities_fictional['Probability of Survival'] = df_probabilities_fictional['Probability of Survival'] * 100
print(df_probabilities_fictional)
## PROBABILITIES FOR ONE FICTIONAL INDIVIDUAL END








