import pandas as pd
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import xgboost as xgb

# read file


trainOrigin = pd.read_csv('~/Documents/DMC16/Aufgabenstellung/orders_train.txt', sep=';')
test = pd.read_csv('~/Documents/DMC16/Aufgabenstellung/orders_class.txt', sep=';')



def preprocess(train, test):

    train = train.drop(['orderID', 'orderDate', 'articleID', 'sizeCode', 'voucherID', 'customerID'], axis=1)
    test = test.drop(['orderID', 'orderDate', 'articleID', 'sizeCode', 'voucherID', 'customerID'], axis=1)

    print("train.info()")
    print(train.info())
    print("remove colorCodeMissingValues")
    train= train[train.rrp.notnull()]
    train= train[train.productGroup.notnull() ]

    test= test[test.rrp.notnull()]
    test= test[test.productGroup.notnull()]

    print("train.info()")
    print(train.info())

    print("test.info()")
    print(test.info())

    x_cat_train = train['paymentMethod'].T.to_dict().values()

    le = preprocessing.LabelEncoder()
    train['paymentMethod'] = le.fit_transform(train['paymentMethod'])
    test['paymentMethod'] = le.transform(test['paymentMethod'])

#   enc = preprocessing.OneHotEncoder()
#   print(enc.fit_transform(train['paymentMethod']))
#    enc.transform(test['paymentMethod'])


    print(train.head())

    return train, test


'''
   print(trainDataset.sizeCode)


'''


def predict(train,label, test):
    print("train")
    clf = xgb.XGBClassifier(max_depth=3, n_estimators=100,learning_rate=0.05).fit(train,label)

    #clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=2,)
    #clf = svm.LinearSVC()
    #clf.fit(train, label)
    print("classify")
    pred = clf.predict(test)
    #test['prediction']=pred
    #predictions = gbm.predict(test)
    return pred

def evaluate(pred,label):
    summe=0
    rowcount=0
    print("len Predictions")
    print(len(pred))
    for row in range(len(pred)):
        summe=summe+abs(pred[row]-label[row])
        rowcount=rowcount+1
    print("summe Fehler",str(summe))
    print("summe rowCount",str(rowcount))

    return summe/rowcount


print("preprocessing")
trainOrigin, test = preprocess(trainOrigin, test)

split=2000000
train = trainOrigin[:split]
test = trainOrigin[split:]
print(train.info())
print(test.info())
label = train[['returnQuantity']].values.ravel()
train=train.drop(['returnQuantity'],axis=1)

labelTest = test[['returnQuantity']].values.ravel()
test=test.drop(['returnQuantity'],axis=1)

print("train and predict")
pred = predict(train,label,test)

print("evaluate")
print(evaluate(pred,labelTest))



# if __name__ == '__main__':
#    main()
