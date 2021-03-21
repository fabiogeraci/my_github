import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 20)
import numpy as np
import seaborn as sns

import requests
import os.path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import jaccard_score, f1_score, log_loss


save_path = os.path.join(os.getcwd(), 'data/loan_train.csv')
print(save_path)
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/' \
      'labs/FinalModule_Coursera/data/loan_train.csv'


def download_url(url_, save_path_, chunk_size=128):
    r = requests.get(url_, stream=True)
    with open(save_path_, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


if not os.path.isfile(save_path):
    download_url(url, save_path)
    print('End Downloading')
else:
    print('File Already Exists')


df = pd.read_csv('data/loan_train.csv')

# Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['effective_date'].dt.dayofweek

printing = False
if printing:
    bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
    g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, 'Principal', bins=bins, ec="k")

    g.axes[-1].legend()
    plt.show()
    #
    bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
    g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
    g.axes[-1].legend()
    plt.show()

    bins = np.linspace(df.terms.min(), df.terms.max(), 5)
    g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, 'terms', bins=bins, ec="k")
    g.axes[-1].legend()
    plt.show()


def prepData(df):
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

    # Convert Categorical features to numerical values
    df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
    df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

    # One Hot Encoding
    df.groupby(['education'])['loan_status'].value_counts(normalize=True)

    # Feature Selection
    Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
    Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
    Feature.drop(['Master or Above'], axis=1, inplace=True)

    # Data Standardization give data zero mean and unit variance (technically should be done after train test split )
    X = preprocessing.StandardScaler().fit(Feature).transform(Feature)
    y = np.array(pd.factorize(df['loan_status'])[0].tolist())

    return X, y


def score_matrix(*args):
    mean_acc_ = args[1]
    j_score_ = args[2]
    f_score_ = args[3]
    l_loss_ = args[4]
    std_acc_ = args[5]
    y_test_ = args[6]
    yhat_ = args[7]
    yhat_probs_ = args[8]
    model_type_ = args[9]

    if args[0] > 0:
        mean_acc_[n - 1] = metrics.accuracy_score(y_test_, yhat_)
        j_score_[n - 1] = jaccard_score(y_test_, yhat_, pos_label=0)
        f_score_[n - 1] = f1_score(y_test_, yhat_, average='weighted')
        if model_type_.startswith('SVC'):
            l_loss_[n - 1] = 0.0
        else:
            l_loss_[n - 1] = log_loss(y_test_, yhat_probs_, normalize=True)
        std_acc_[n - 1] = np.std(yhat_ == y_test_) / np.sqrt(y_test_.shape[0])

    else:
        mean_acc_ = metrics.accuracy_score(y_test_, yhat_)
        j_score_ = jaccard_score(y_test_, yhat_, pos_label=0)
        f_score_ = f1_score(y_test_, yhat_, average='weighted')

        if model_type_.startswith('SVC'):
            l_loss_ = 0
        else:
            l_loss_ = log_loss(y_test_, yhat_probs_, normalize=True)
        std_acc_ = np.std(yhat_ == y_test_) / np.sqrt(y_test_.shape[0])

    return mean_acc_, j_score_, f_score_, l_loss_, std_acc_


X, y = prepData(df)

# print(X[0:5])
# print(y[0:5])

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# K Nearest Neighbor(KNN)
print('---------K Neighbors Classifier--------------')
Ks = 10

mean_acc = np.zeros((Ks - 1))
j_score = np.zeros((Ks - 1))
f_score = np.zeros((Ks - 1))
l_loss = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n, algorithm='auto').fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    yhat_probs = neigh.predict_proba(X_test)

    model_type = str(neigh)

    mean_acc, j_score, f_score, l_loss, std_acc = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc,
                                                               y_test, yhat, yhat_probs, model_type)

print("The best accuracy was with {:.3f}".format(mean_acc.max()), "with k=", mean_acc.argmax() + 1)
print("Max j_score was with {:.3f}".format(j_score.max()), "with k=", j_score.argmax() + 1)
print("Max f_score was with {:.3f}".format(f_score.max()), "with k=", f_score.argmax() + 1)
print("Max l_loss was with {:.3f}".format(l_loss.max()), "with k=", l_loss.argmax() + 1)
print("Max standard accuracy was with {:.3f}".format(std_acc.max()), "with k=", std_acc.argmax() + 1)

mean_acc_kn = mean_acc.argmax() + 1
print('---------------------------------------------')

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

defaultTree = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features='sqrt', random_state=4)

mean_acc = np.zeros(1)
j_score = np.zeros(1)
f_score = np.zeros(1)
l_loss = np.zeros(1)
std_acc = np.zeros(1)

defaultTree.fit(X_train, y_train)
yhat = defaultTree.predict(X_test)
yhat_probs = defaultTree.predict_proba(X_test)

model_type = 'defaultTree'

n = 0
mean_acc, j_score, f_score, l_loss, std_acc = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc,
                                                           y_test, yhat, yhat_probs, model_type)
print('---------Decision Tree Classifier--------------')
print("The best accuracy was with {:.3f}".format(mean_acc))
print("Max j_score was with {:.3f}".format(j_score))
print("Max f_score was with {:.3f}".format(f_score))
print("Max l_loss was with {:.3f}".format(l_loss))
print("Max standard accuracy was with {:.3f}".format(std_acc))
print('-----------------------------------------------')
#
# # Support Vector Machine
# """
# Kernels to be tested
#
# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid
# """

from sklearn import svm
#
list_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

mean_acc = np.zeros(len(list_kernels))
j_score = np.zeros(len(list_kernels))
f_score = np.zeros(len(list_kernels))
l_loss = np.zeros(len(list_kernels))
std_acc = np.zeros(len(list_kernels))

for n, kernel in enumerate(list_kernels):
    print(n + 1, kernel)
    clf_svm = svm.SVC(C=100, kernel=kernel, probability=True)

    model_type = str(clf_svm).split(',')[0]

    # Train Model and Predict
    clf_svm.fit(X_train, y_train)
    yhat = clf_svm.predict(X_test)
    yhat_probs = clf_svm.predict_proba(X_test)

    n = n + 1
    mean_acc, j_score, f_score, l_loss, std_acc = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc,
                                                               y_test, yhat, yhat_probs, model_type)
print('---------Support Vector Machine--------------')
print("The best accuracy was with {:.3f}".format(mean_acc.max()), "with k=", mean_acc.argmax() + 1)
print("Max j_score was with {:.3f}".format(j_score.max()), "with k=", j_score.argmax() + 1)
print("Max f_score was with {:.3f}".format(f_score.max()), "with k=", f_score.argmax() + 1)
print("Max l_loss was with {:.3f}".format(l_loss.max()), "with k=", l_loss.argmax() + 1)
print("Max standard accuracy was with {:.3f}".format(std_acc.max()), "with k=", std_acc.argmax() + 1)
print('---------------------------------------------')
mean_acc_svm = mean_acc.argmax() + 1

# Logistic Regression
print('---------Logistic Regression--------------')
from sklearn.linear_model import LogisticRegression

l1_ratio = 0.5  # L1 weight in the Elastic-Net regularization
C = 0.1
tol = 0.01
solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']

mean_acc = np.zeros(len(solvers))
j_score = np.zeros(len(solvers))
f_score = np.zeros(len(solvers))
l_loss = np.zeros(len(solvers))
std_acc = np.zeros(len(solvers))

for n, solver in enumerate(solvers):
    print(n + 1, solver)
    if solver == 'saga':
        LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='elasticnet', l1_ratio=l1_ratio, tol=tol)
    elif solver == 'lbfgs' or solver == 'newton-cg' or solver == 'sag':
        LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='l2', tol=tol)
    else:
        LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='l2', tol=tol)
    LR.fit(X_train, y_train)
    yhat = LR.predict(X_test)
    yhat_probs = LR.predict_proba(X_test)

    model_type = str(LR).split(',')[0]

    n = n + 1
    mean_acc, j_score, f_score, l_loss, std_acc = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc, y_test,
                                                               yhat, yhat_probs, model_type)

print("The best accuracy was with {:.3f}".format(mean_acc.max()), "with k=", mean_acc.argmax() + 1)
print("Max j_score was with {:.3f}".format(j_score.max()), "with k=", j_score.argmax() + 1)
print("Max f_score was with {:.3f}".format(f_score.max()), "with k=", f_score.argmax() + 1)
print("Max l_loss was with {:.3f}".format(l_loss.max()), "with k=", l_loss.argmax() + 1)
print("Max standard accuracy was with {:.3f}".format(std_acc.max()), "with k=", std_acc.argmax() + 1)

save_path = os.path.join(os.getcwd(), 'data/loan_test.csv')
print(save_path)

url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv'

if not os.path.isfile(save_path):
    download_url(url, save_path)
    print('End Downloading')
else:
    print('File Already Exists')

df_new = pd.read_csv('data/loan_test.csv')
df_new.head()

df_new['due_date'] = pd.to_datetime(df_new['due_date'])
df_new['effective_date'] = pd.to_datetime(df_new['effective_date'])
df_new['dayofweek'] = df_new['effective_date'].dt.dayofweek
print(df_new['loan_status'].value_counts())

X_new, y_new = prepData(df_new)
print('---------K Neighbors Classifier--------------')
# Train Model and Predict KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=mean_acc_kn, algorithm='auto').fit(X, y)
yhat = neigh.predict(X_new)
yhat_probs = neigh.predict_proba(X_new)

model_type = str(neigh).split(',')[0]

n = 0
mean_acc_kn, j_score_kn, f_score_kn, l_loss_kn, std_acc_kn = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc,
                                                                          y_new, yhat, yhat_probs, model_type)


print('---------------------------------------------')
#
# Decision Tree
yhat = defaultTree.predict(X_new)
yhat_probs = defaultTree.predict_proba(X_new)

model_type = str(defaultTree).split(',')[0]

n = 0
mean_acc_dt, j_score_dt, f_score_dt, l_loss_dt, std_acc_dt = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc,
                                                                          y_new, yhat, yhat_probs, model_type)
# Support Vector Machine
# Train Model
clf_svm = svm.SVC(C=100, kernel='poly', probability=True)
clf_svm.fit(X_train, y_train)
model_type = 'SVC'

# Predict
yhat = clf_svm.predict(X_new)
yhat_probs = clf_svm.predict_proba(X_new)

n = 0
mean_acc_svc, j_score_svc, f_score_svc, l_loss_svc, std_acc_svc = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc, y_new, yhat,
                                                                               yhat_probs, model_type)

# Logistic Regression
solver = 'liblinear'

if solver == 'saga':
    LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='elasticnet', l1_ratio=l1_ratio, tol=tol)
elif solver == 'lbfgs' or solver == 'newton-cg' or solver == 'sag':
    LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='l2', tol=tol)
else:
    LR = LogisticRegression(C=C, solver=solver, random_state=4, penalty='l2', tol=tol)


LR.fit(X_train, y_train)
yhat = LR.predict(X_new)
yhat_probs = LR.predict_proba(X_new)

model_type = 'LR'

n = 0
mean_acc_lr, j_score_lr, f_score_lr, l_loss_lr, std_acc_lr = score_matrix(n, mean_acc, j_score, f_score, l_loss, std_acc, y_new, yhat,
                                                                          yhat_probs, model_type)

print('-----------------------------------------------')

result_dc = {'Algorithm':['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'],
             'Jaccard':["{:.5f}".format(j_score_kn),"{:.5f}".format(j_score_dt),"{:.5f}".format(j_score_svc), "{:.5f}".format(j_score_lr)],
             'F1-score':["{:.5f}".format(f_score_kn),"{:.5f}".format(f_score_dt),"{:.5f}".format(f_score_svc), "{:.5f}".format(f_score_lr)],
             'LogLoss':['NA','NA','NA', "{:.5f}".format(l_loss_lr)]}

print(pd.DataFrame(result_dc))
