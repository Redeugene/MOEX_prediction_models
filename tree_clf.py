'''import pandas as pd
tickers = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\container.txt', header = None)
tickers['target'] = pd.cut(tickers[21],[-100, -20, -10, -5, 0, 5, 10, 15, 20, 500], labels = [-4, -3, -2, -1, 1, 2, 3, 4, 5])
x = tickers.drop(columns = [21,22, 'target'])
y = tickers['target']
print(y)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(random_state=0, max_depth = 2, criterion = 'entropy')
clf.fit(x,y)
#print(clf.feature_importances_)
#print(clf.predict(x.iloc[200].values.reshape(1,-1)))
#print(y.iloc[200])
test = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\test.txt', header = None)
test['target'] = pd.cut(test[21],[-100, -20, -10, -5, 0, 5, 10, 15, 20, 500], labels = [-4, -3, -2, -1, 1, 2, 3, 4, 5])
x_test = test.drop(columns = [21,22, 'target'])
y_test = test['target']
print(clf.predict(x_test.iloc[2].values.reshape(1,-1)))
print(y_test.iloc[2])
tree.plot_tree(clf, fontsize = 7)
#print(tickers.describe())
#y.plot.hist()'''


'''import pandas as pd
tickers = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\container.txt', header = None)
x = tickers.drop(columns = [21,22])
y = tickers[21]
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
#clf = MLPRegressor(random_state=0, max_iter = 500)
clf = DecisionTreeRegressor(max_depth  = 100)
clf.fit(x,y)
#print('Weights:', clf.feature_importances_)
#print(clf.predict(x.iloc[200].values.reshape(1,-1)))
#print(y.iloc[200])
test = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\test.txt', header = None)
x_test = test.drop(columns = [21,22])
y_test = test[21]
a = pd.DataFrame(columns = ['Predict of tree', 'Real value', 'Predict of weighted tree model'], index = range(len(x_test)))
importances = clf.feature_importances_
for i in range(len(x_test)):
    a.iloc[i] = [clf.predict(x_test.iloc[i].values.reshape(1,-1))[0], y_test.iloc[i], np.dot(np.array(importances), np.array(x_test.iloc[i]))]
predicts, reals, pred_weighted = [], [], []
for i in a['Predict of weighted tree model']:
    if i < 0:
        pred_weighted.append(0)
    else:
        pred_weighted.append(1)
for j in a['Predict of tree']:
    if j < 0:
        predicts.append(0)
    else:
        predicts.append(1)
for k in a['Real value']:
    if k < 0:
        reals.append(0)
    else:
        reals.append(1)
#print("Predicts: ", predicts)
#print("Real values: ", reals)
#print("Predict of weighted tree model: ", pred_weighted)
ttff = []
for m,n in zip(predicts, reals):
    if m == n:
        ttff.append(1)
ttff_weight = []
for m,n in zip(pred_weighted, reals):
    if m == n:
        ttff_weight.append(1)
print('Accuracy tree = ', len(ttff)/len(predicts)*100)
print('Accuracy weighted tree= ', len(ttff_weight)/len(predicts)*100)
#tree.plot_tree(clf)
#print(tickers.describe())
#y.plot.hist(bins = 20)
#importances = clf.feature_importances_
#print("Predict of weighted tree model: ", np.dot(np.array(importances), np.array(x_test.iloc[3])))
clf_regr = LinearRegression()
clf_regr.fit(x,y)
#print('Predict of lin regression: ', clf_regr.predict(x_test.iloc[2].values.reshape(1,-1)))
from sklearn.metrics import confusion_matrix
import seaborn
seaborn.heatmap(confusion_matrix(np.array(predicts), np.array(reals)), annot = True, fmt = 'd')
#print(a)'''

import pandas as pd
tickers = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\container.txt', header = None)
x = tickers.drop(columns = [21,22])
y = tickers[21]
print(y)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
#clf = MLPRegressor(random_state=0, max_iter = 500)
#params = {'hidden_layer_sizes': [(i,j) for i in range(10) for j in range(10)]}
#clf = MLPRegressor('random_state' : 0, 'max_iter':1000)
#clf.fit(x,y)
#print('Weights:', clf.feature_importances_)
#print(clf.predict(x.iloc[200].values.reshape(1,-1)))
#print(y.iloc[200])
test = pd.read_csv(r'C:\\Users\\pc\\Downloads\\json_date\\test.txt', header = None)
x_test = test.drop(columns = [21,22])
y_test = test[21]
#(4,3) alpha = 1, beta_1 = 0.85, beta_2 = 0.85 (4,9) (6,4)
#importances = clf.feature_importances_
for length, number in zip([4], [9]):
        clf = MLPRegressor(random_state = 0, max_iter=10000, hidden_layer_sizes = (length, number), alpha = 1, beta_1 = 0.85, beta_2 = 0.85)
        clf.fit(x,y)
        a = pd.DataFrame(columns = ['Predict of tree', 'Real value'], index = range(len(x_test)))
        for i in range(len(x_test)):
            a.iloc[i] = [clf.predict(x_test.iloc[i].values.reshape(1,-1))[0], y_test.iloc[i]]
        predicts, reals = [], []
        for j in a['Predict of tree']:
            if j < 0:
                predicts.append(0)
            else:
                predicts.append(1)
        for k in a['Real value']:
            if k < 0:
                reals.append(0)
            else:
                reals.append(1)
        print("Predicts: ", predicts)
        print("Real values: ", reals)
        ttff = []
        for m,n in zip(predicts, reals):
            if m == n:
                ttff.append(1)
        print('Accuracy model = ', len(ttff)/len(predicts)*100)


#tree.plot_tree(clf)
#print(tickers.describe())
#y.plot.hist(bins = 20)
#importances = clf.feature_importances_
#print("Predict of weighted tree model: ", np.dot(np.array(importances), np.array(x_test.iloc[3])))
#clf_regr = LinearRegression()
#clf_regr.fit(x,y)
#print('Predict of lin regression: ', clf_regr.predict(x_test.iloc[2].values.reshape(1,-1)))
from sklearn.metrics import confusion_matrix
import seaborn
seaborn.heatmap(confusion_matrix(np.array(predicts), np.array(reals)), annot = True, fmt = 'd')
print(a)
tp = 39
fn = 1
fp = 6
tn = 6
print('Recall = ', tp/(tp+fn))
print('Precision = ', tp/(tp+fp))

