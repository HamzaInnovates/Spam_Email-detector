
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("combined_data.csv")
# train_test_split
split= StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
for train_index, test_index in split.split(df,df["label"]):
    train=df.loc[train_index]
    test=df.loc[test_index]
    
#creating object of count vectorizer and transforming the input data
cv= CountVectorizer()
X_train = cv.fit_transform(train['text'])  
X_test = cv.transform(test['text'])

Y_train = train['label']
Y_test = test['label']

#training models
clf1 = MultinomialNB()
clf1.fit(X_train, Y_train)

# claculating predictions and accuracy_score
pred_1 = clf1.predict(X_test)
acc_1 = accuracy_score(Y_test, pred_1)

# printing confusion_matrix and accuracy_score
print("Accuracy of MultinomialNB:", acc_1 * 100)
print("Confusion matrix of MultinomialNB: \n", confusion_matrix(Y_test, pred_1))


# function to calculate root mean_squared_error
def rmse(mse):
    return np.sqrt(mse)
#mean
mse1 = mean_squared_error(Y_test, pred_1)


#Calculating and printing root mean_squared_error
rmse1 = rmse(mse1)

print(f"RMSE for MultinomialNB : {rmse1}")

X = cv.fit_transform(df['text'].astype(str)) 
Y = df['label']

#Calculating and printing cross_val_score
scores1 = cross_val_score(clf1, X, Y, cv=4, scoring='accuracy')

print(f"Cross-Validation Accuracy Score for MultinomialNB: {scores1}")
print(f"Mean Accuracy: {scores1.mean() * 100}\n")

# Visualize the distribution of spam and ham emails
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Spam and Ham Emails')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['0', '1'], rotation=0)
plt.show()

#test email

# email = [""]


# email = cv.transform(email)

# predictions = clf1.predict(email) 
# for i, pred in enumerate(predictions):
#     result = "Spam" if pred == 1 else "Not Spam"
#     print(f"Email {i+1}: {result}")
