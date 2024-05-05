# ML-Project-7-Ad-Click-Prediction.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as  plt
import plotly.express as px

df=pd.read_csv("/content/drive/MyDrive/advertising.csv")
Overview

df.head()

df.shape

df.info()

df.describe()

df.describe(include='O')

# Ad Topic Line are all unique so we will drop it

# City mostly doesn't change so we will drop it too `

df=df.drop(columns=['City','Ad Topic Line'],axis=1)
df.head()

df['Date']=pd.to_datetime(df['Timestamp'])

df=df.drop(columns=['Timestamp'],axis=1)
df.head()

df['Date'].min(),df['Date'].max()

# All Data is in the Same Year so year is not important

df.isna().sum()

df.duplicated().sum()

df.select_dtypes('object').nunique()

df.describe()

# Features Distribution

num_features = df.select_dtypes('number').columns.drop('Clicked on Ad')
fig,ax = plt.subplots(1,5,figsize=(25,5))
for i,col in enumerate(num_features):
    sns.histplot(data=df,x=col,ax=ax[i],kde=True)
    ax[i].set_title(f'{col} Distribution')

plt.figure(figsize=(50,15))
sns.countplot(data=df,x='Country')
plt.xticks(rotation=60);

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=42)
#Target Distribution

sns.countplot(data=df,x='Clicked on Ad');

# Target Classes are balanced.

# Features Correlation with Target

fig,ax = plt.subplots(1,5,figsize=(25,5))
for i,col in enumerate(num_features):
    sns.histplot(data=df,x=col,ax=ax[i],kde=True,hue='Clicked on Ad')
    ax[i].set_title(f'{col} Distribution')

plt.figure(figsize=(12,6))
sns.histplot(data=df,x='Age')

sns.jointplot(x=df['Age'],y=df['Area Income'],height=12)

sns.jointplot(x=df['Daily Time Spent on Site'],y=df['Daily Internet Usage'],height=12)

sns.jointplot(x=df['Age'],y=df['Daily Time Spent on Site'],kind='kde',height=12)

sns.pairplot(data=df,hue='Clicked on Ad')

# We can see that daily less internet usage tends to click on ad more.

#Lets see Click on Ad features based on Sex
plt.figure(figsize=(10,6))
sns.countplot(x='Clicked on Ad',data=df,hue='Male',palette='coolwarm')

# Female tends to click more on Ads!

#Distribution of top 12 country's df clicks based on Sex
plt.figure(figsize=(15,6))
sns.countplot(x='Country',data=df[df['Clicked on Ad']==1],order=df[df['Clicked on Ad']==1]['Country'].value_counts().index[:12],hue='Male',
              palette='viridis')
plt.title('Ad clicked country distribution')
plt.tight_layout()

# Most are developing countries and females are the active contributors.

df

#Now we shall introduce new columns Hour,Day of Week, Date, Month from timestamp
df['Hour']=df['Date'].apply(lambda time : time.hour)
df['DayofWeek'] = df['Date'].apply(lambda time : time.dayofweek)
df['Month'] = df['Date'].apply(lambda time : time.month)
df['Date'] = df['Date'].apply(lambda t : t.date())

#Hourly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Hour',data=df[df['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked hourly distribution')

# As we can see with uneven daytime frequency, females are the main contributor exceeding males several hours.

#Daily distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='DayofWeek',data=df[df['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked daily distribution')

#Most of the Days Ladies click ad more than Males except Wednesdays and Thursdays.

#Monthly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Month',data=df[df['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked monthly distribution')

#Throughout the Year Ladies click on Ad the most except month of March.

#Now we shall group by date and see the
plt.figure(figsize=(15,6))
df[df['Clicked on Ad']==1].groupby('Date').count()['Clicked on Ad'].plot()
plt.title('Date wise distribution of Ad clicks')
plt.tight_layout()

#Top df clicked on specific date
df[df['Clicked on Ad']==1]['Date'].value_counts().head(5)

#On 14th February 2016 we see most (8) clicks on ad. So Valentine Day is the best selling day for the Company's Ad.

df

#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.distplot(df['Age'],kde=False,bins=40)

#Most of them are around 30 years. But is this age group clicking most on Ad?

#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.swarmplot(x=df['Clicked on Ad'],y= df['Age'],data=df,palette='coolwarm')
plt.title('Age wise distribution of Ad clicks')

#As its clear from above that around 40 years population are the most contributor to ad clickings and not around 30 years.

#As we can see people around 30 years population devote lot of their time on internet and on the site, but they don't click on Ads that frequent. Comapred to them, around 40 years population spend a bit less time but click on Ads more.

#Lets see the distribution who clicked on Ad based on area income of sex
plt.figure(figsize=(10,6))
sns.violinplot(x=df['Male'],y=df['Area Income'],data=df,palette='viridis',hue='Clicked on Ad')
plt.title('Clicked on Ad distribution based on area distribution')

#Both Males and Females with Area income less than 50k are main customers of Ad. As almost all whose income more than 60k are not interested on clicking on Ad.

#Thus in conclusion, mostly around 40 years Female within income group less than 50k in developing countries are the main consumers of Ad, clicking unevenly throughout the day and mostly during Fridays and Sundays

df

df1=df.drop(['Country','Date'],axis=1)

sns.heatmap(df1.corr(),annot=True,cmap='Blues');

#Now, with these observations, let’s build our logistic regression model to predict ad click probability.

#Model Building We are going to use the logistic regression model because it’s easy to implement and very efficient to train. I’d suggest taking logistic regression as a benchmark, experimenting with more complex algorithms, and checking the improvement rooms.

#We will consider the following features for our model because, from the correlation matrix, we saw that these are the influencing factors for predicting advertisement clicks.

#Daily Time Spent on Site

#Age
#Area Income
#Daily Internet Usage
#Male (gender)

#Now, you know the steps. First, select the feature columns and the label then split the dataset into random train and test subsets with scikit-learn’s handy train_test_split function. After that, build and train a machine learning model. For us, it is a logistic regression model.

#Note: We don’t need to standardize our dataset before training the model because logistic regression is not that sensitive to the different scales of features.

#Lets take country value as dummies
country= pd.get_dummies(df['Country'],drop_first=True)

#Now lets join the dummy values
df = pd.concat([df,country],axis=1)

#Fitting Data

#Training Model

def model_evaluation(model, X_test, y_test, color='Blues'):
   """
    This function evaluates the performance of a trained model on the test set.

    Args:
        model: The trained machine learning model.
        X_test: The test data.
        y_test: The true labels for the test data.
        color: The color map to be used for plotting the confusion matrix.

    Returns:
        None
    """
    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Classification report
    print('--------------------------------------------------------------')
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))
    print('--------------------------------------------------------------')
    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cmap=color)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def model_tunning(model,X_train,y_train,parameters):
    '''
    This function receives a model, tunes it using GridSearchCV, prints the best parameters,
    and returns the best estimator.

    Args:
        model: The machine learning model to be tuned.
        X_train: The training data.
        y_train: The target labels.
        parameters: The grid of hyperparameters to be tuned.

    Returns:
        best_estimator: The best estimator found during grid search.
    '''
    grid_search = GridSearchCV(estimator=model,param_grid=parameters,cv=5,scoring='f1')
    grid_search.fit(X_train,y_train)
    print("Best parameters are: ",grid_search.best_params_)
    print('Mean cross-validated f1 score of the best estimator is: ',grid_search.best_score_)
    return grid_search.best_estimator_

def cross_validation(model,X_train,y_train,n):

    """
    This function is used to validate the model across multiple stratified splits.
    Args:
        model: The machine learning model to be evaluated.
        X_train: The training data.
        y_train: The target labels.
    Returns:
        None
    """
    splits = StratifiedKFold(n_splits=n,random_state=42,shuffle=True)
    validation_scores = cross_val_score(model,X_train,y_train,cv=splits,scoring='f1')
    print('Scoring Metric: f1')
    print('Cross Validation Scores: ',validation_scores)
    print('Scores Mean: ',validation_scores.mean())
    print('Scores Standard Deviation: ',validation_scores.std())
    print('--------------------------------------------------------------')

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV , StratifiedKFold , cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix , classification_report , roc_auc_score , roc_curve ,f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier , RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier

import pickle

# Preparing & Fitting Data
features = df.drop(['Country','Clicked on Ad','Date'],axis=1)
target = df["Clicked on Ad"]

# Training Model
ss = StandardScaler()

log = LogisticRegression()
dec = DecisionTreeClassifier()
ran = RandomForestClassifier()
nn = MLPClassifier()

log_pipe = make_pipeline(ss,log)
dec_pipe = make_pipeline(ss,dec)
ran_pipe = make_pipeline(ss,ran)
nn_pipe = make_pipeline(ss,nn)

kf = KFold(n_splits=10, shuffle=True, random_state = 1)

log_results = cross_val_score(log_pipe,features,target, cv=kf, scoring = 'accuracy')
dec_results = cross_val_score(dec_pipe,features, target, cv=kf, scoring = 'accuracy')
ran_results = cross_val_score(ran_pipe, features, target, cv=kf, scoring = 'accuracy')
nn_results = cross_val_score(nn_pipe, features, target, cv=kf, scoring = 'accuracy')

pd.DataFrame({'Algorithm':['Logistic Regression','Decision Tree','Random Forest','Neural Network'],
             'K-Fold Accuracy':[log_results.mean(),dec_results.mean(),
                               ran_results.mean(),nn_results.mean()]})

#Let's use another technique for assessing accuracy:

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2)

log = LogisticRegression()
dec = DecisionTreeClassifier()
ran = RandomForestClassifier()
nn = MLPClassifier()

models = [log,dec,ran,nn]

for model in models:
    model.fit(X_train,y_train)

for model in models:
    print(model)
    print(classification_report(y_test,model.predict(X_test)))
    print("")

LogisticRegression()

DecisionTreeClassifier()

RandomForestClassifier()

MLPClassifier()

#We'll use Random Forest for now, it has consistently the best results.

model = ran

for model in models:
    model.fit(X_test,y_test)

for model in models:
    print(model)
    print(classification_report(y_test,model.predict(X_test)))
    print("")

LogisticRegression()

DecisionTreeClassifier()

RandomForestClassifier()

MLPClassifier()

logpredictions = log.predict(X_test)

#logpredictions

#For better parameters we will apply GridSearch

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000]}

#Logistic Regression

grid_log= GridSearchCV(LogisticRegression(),param_grid,refit=True, verbose=2)

grid_log.fit(X_train,y_train)

grid_log.best_estimator_

pred_log= grid_log.predict(X_test)

pred_log

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test,logpredictions))
print(classification_report(y_test,logpredictions))

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))

#Let's compare it with other Classification Models!

#DecisionTreeClassifier

from sklearn.model_selection import cross_validate
cross_validate(DecisionTreeClassifier(),X_train,y_train)

param_grid_dec = {
    'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split
    'max_depth': np.arange(1,30),  # The maximum depth of the tree
    'min_samples_split': np.arange(2,10),  # The minimum number of samples required to split an internal node
}
dec_tree = model_tunning(DecisionTreeClassifier(),X_train,y_train,param_grid_dec)

model_evaluation(dec_tree,X_test,y_test,'Reds')

y_pred = dec_tree.predict(X_test)

y_pred

#Random Forest

from sklearn.ensemble import RandomForestClassifier

regr_model = RandomForestClassifier(min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      max_features='auto')

regr_model.fit(X_train, y_train)

pred_dec = regr_model.predict(X_test)

y_pred = regr_model.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = regr_model.predict(X_test)

y_pred

#Support Vector Model

from sklearn.svm import SVC
svc= SVC(gamma='scale')

svc.fit(X_train,y_train)

#For better parameters we need to apply GridSearch

param_grid = {'C': [0.1,1,10,100,1000,5000]}

grid_svc= GridSearchCV(SVC(gamma='scale',probability=True),param_grid,refit=True,verbose=2)

grid_svc.fit(X_train,y_train)

grid_svc.best_estimator_

pred_svc= grid_svc.predict(X_test)
print(confusion_matrix(y_test,pred_svc))
print(classification_report(y_test,pred_svc))

pred_svc

#It does perform better but a slight less than Random Forest Model.

#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

nb_predict = nb.predict(X_test)
print(classification_report(y_test,nb_predict))

nb_predict

#KNN Classifier

cross_validation(KNeighborsClassifier(),X_train,y_train,5)

param_grid_knn = {
    'n_neighbors':np.arange(1,21,2)
}
knn = model_tunning(KNeighborsClassifier(),X_train,y_train,param_grid_knn)

model_evaluation(knn,X_test,y_test,'Blues')

knn_predict = knn.predict(X_test)
print(classification_report(y_test,knn_predict))

knn_predict

#XGBoost Classifier

xgb_clf = XGBClassifier(n_estimators = 100 ,n_jobs = -1 ,random_state = 42)

cross_validation(xgb_clf,X_train,y_train,5)

xgb_clf.fit(X_train,y_train)

model_evaluation(xgb_clf,X_test,y_test,'Reds')

xgb_clf_predict = xgb_clf.predict(X_test)
print(classification_report(y_test,xgb_clf_predict))

xgb_clf_predict

#AdaBoost Classifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=42)

cross_validation(ada_clf,X_train,y_train,5)

ada_clf.fit(X_train,y_train)

model_evaluation(ada_clf,X_test,y_test,'Greens')

#Conclusion

#The best estimator is naive bias and XGB with accurracy of 0.98 on test set. Daily internet usage and Daily time spent on site have very strong negative correlation with the target. Age and Area income have moderate positive correlation with the target. Gender has no effect on the target.

#Based on the above we see that XGB Classifier and Naive Bayes Classifier perfomed the best among the above.

#naive bias and XGB have the best accuracy with 0.98 on test data and f1 score=0.94 and f1 score=0.98

with open("best_estimator.pkl",'wb') as file:
    pickle.dump(svc,file)
