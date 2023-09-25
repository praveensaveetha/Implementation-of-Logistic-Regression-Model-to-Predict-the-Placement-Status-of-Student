# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Praveen V 
RegisterNumber:  212222040121

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Placement Data:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/301bcf7c-159c-4f3a-88e9-41d7ab9dc5c0)

### Salary Data:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/c0e68361-4617-44a7-89bb-42791ab3628c)

### Checking the null() function:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/551f0c2d-925e-45c0-a929-bef5ad4768d5)

### Data Duplicate:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/9fe1aa73-12bc-4438-add0-069645f1f8b1)

### Print Data:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/8b3f8b8d-be42-44ff-9499-fc1ec8b64565)

### Data-status:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/fab32df9-43a0-4a86-9800-ad429f4a44dd)

### y_prediction array:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/6f02fa7a-444d-4ba6-9579-e67376b7349c)

### Accuracy value:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/4ee1e69b-2b7e-40ae-9032-2eccee9b813b)

### Confusion array:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/a8876934-ff4a-4876-a086-cad987d12e9f)

### Classification report:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/da1b2d12-14d5-4a8a-8200-8af73d4b2b4e)

### Prediction of LR:
![image](https://github.com/praveensaveetha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560117/058be5c8-5701-4ac4-bc35-c74bc34fe95a)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
