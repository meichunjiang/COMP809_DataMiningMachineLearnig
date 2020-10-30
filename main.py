
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,auc
from sklearn.metrics import roc_curve,roc_auc_score, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#import seaborn as sn

""" answer table(function--->Question)
#confusionMatrix                 #part B-Q1.1
#plotTrainProbForClassifier      #part B-Q1.2

#part B-Q1.3
For every sample, the predicted prob is a probability distribution in every class, this is the predicted 
probabilities vector. The class corresponding to the maximum value in this vector is the predicted class.
In the predicted probability matrix, for every class, sum the total sample number of this class marked as M, 
and sum all these sample's predicted probability marked as N. The value of N/M is the Average Aggregate 
probability for this class.

#predicteProbStatistic      #part B-Q1.4

#part B-Q1.5 a,b
Just use the average aggregate probability as the conditional probability.
The max probability in each row means predict the result of sample, 
  sometimes the result is more likely different in various classifiers.
Compare with two probabilities in different classifier - DTree and MLP
method-> the max pro multiply the average aggregate probability respectively
result-> choose higher probability in same sample.
It will be helpful to obviously increase accuracy because choose higher accuracy result in two classifiers.

#predictByTwoClfConditProb    part B-Q1.6

"""


'''

class_mapping = {
        's ': 3,
        'h ': 2,
        'd ': 1,
        'o ': 0}

def getData():
    path = r'/Users/chunjiangmei/Documents/forest.xlsx' #r'../../Iris.xlsx'  # should change the path accordingly

    rawdata = pd.read_excel(path)  # pip install xlrd
    print("data summary")
    #print(rawdata.describe())
    nrow, ncol = rawdata.shape  # 523 28
    print('nrow=', nrow, 'ncol=', ncol)
    #print(rawdata.head())

    rawdata = preProcessData(rawdata)
    return rawdata

def preProcessData(data):
    # print(rawdata.isnull().sum())
    #print(data['class'])
    data['class'] = data['class'].map(class_mapping)
    #print(data['class'])
    #print(data.head())
    return data

def showCorr(data):
    print ("\n correlation Matrix")
    print (data.corr())
    data.hist()
    plt.show()

def scatter_matrix(data):
    pd.plotting.scatter_matrix(data, figsize=[8, 8])
    plt.show()

def boxplot(data):
    # boxplot
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data.values)
    ax.set_xticklabels(['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width', 'Class'])
    plt.show()

def DecisionTree(pred_train, pred_test, tar_train, tar_test):
    print('*' * 60, DecisionTree.__name__)
    classifier = DecisionTreeClassifier(max_depth=4, criterion='entropy', min_samples_split=5)  # configure the classifier
    classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
    predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set

    #print('decision tree depth:',classifier.get_depth())

    confusionMatrix(classifier, predictions, tar_test)  #part B-Q1.1

    print("Accuracy score of our model with Decision Tree:", accuracy_score(tar_test, predictions))
    # precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
    # print("Precision score of our model with Decision Tree :", precision)
    # recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
    # print("Recall score of our model with Decision Tree :", recall)

    return classifier, getProbabilitiesModel(classifier, pred_test)  #part B-Q1.2

def MLPClassifierModel(pred_train, pred_test, tar_train, tar_test):
    print('*' * 60, MLPClassifierModel.__name__)
    clf = MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,),learning_rate_init=0.01,max_iter=600)
    clf.fit(pred_train, np.ravel(tar_train, order='C'))
    predictions = clf.predict(pred_test)
    print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions))
    scores = cross_val_score(clf, pred_test, tar_test, cv=10)
    print("Accuracy score of our model with MLP under cross validation :", scores.mean())
    confusionMatrix(clf, predictions, tar_test)
    return clf, getProbabilitiesModel(clf,pred_test)


def confusionMatrix(clf,predictions, targets):
    print("*"*60, confusionMatrix.__name__)
    results = confusion_matrix(targets, predictions)
    print('Confusion Matrix :')
    print(clf.classes_)
    print(results)
    print('Accuracy Score :', accuracy_score(targets, predictions))
    #print('Report : ')
    #print(classification_report(targets, predictions))
    #plotConfusionMatrix(results)
    pass

def getProbabilitiesModel(clf, pred_test,NoSamples=0):
    assert(clf != None)

    prob = clf.predict_proba(pred_test)  # obtain probability scores for each sample in test set
    # logProb = classifier.predict_log_proba(pred_test) #obtain log-probability

    #print('pred_test.shape = ', pred_test.shape)
    #print('prob.shape=', prob.shape)
    #print(prob[NoSamples])

    # print('logProb.shape=', logProb.shape)
    # print(logProb[:10])
    return prob  # the first sample probablities for every class

def plotConfusionMatrix(matrix, classes=['0','1','2','3']):
    df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

def predictByTwoClfConditProb(trainProbDTree, trainProbMLP, probAverageDTree, probAverageMLP, classess):
    print('*' * 10, predictByTwoClfConditProb.__name__, 'predict by 2 clf conditional prob')
    print(trainProbDTree.shape)
    print(trainProbMLP.shape)
    print(classess)
    print('probAverageDTree = ', probAverageDTree)
    print('probAverageMLP = ', probAverageMLP)

    predicts=[] #np.zeros((len(trainProbDTree),1))
    smaples = 0
    for probDTree,probMLP in zip(trainProbDTree,trainProbMLP):
        #print(probDTree,probMLP)
        i = list(probDTree)
        max_probDTree = max(i)
        classIdDTree = i.index(max_probDTree)

        j = list(probMLP)
        max_probMLP = max(j)
        classIdMLP = j.index(max_probMLP)

        classId = 0
        if classIdDTree == classIdMLP:
            classId = classIdDTree
        else:
            P1 = max_probDTree*probAverageDTree[classIdDTree]
            P2 = max_probMLP*probAverageMLP[classIdMLP]
            if P1 > P2:
                classId = classIdDTree
                print("samples in test set: ", smaples, 'change select from MLP class:',classess[classIdMLP],
                      'to DTree class:',classess[classIdDTree])
            else:
                classId = classIdMLP
                print("samples in test set: ", smaples, 'change select from DTree class:', classess[classIdDTree],
                      'to MLP class:', classess[classIdMLP])


        predict = classess[classId]
        #print(classId,predict)
        predicts.append(predict)
        smaples+=1

    return predicts


def plotTrainProbForClassifier(DTreeProb,MLPProb,classes):
    print('*' * 10, plotTrainProbForClassifier.__name__,'first sample prob in 2 classifiers')
    # print(DTreeProb.shape)
    # print(DTreeProb)
    # print(MLPProb.shape)
    # print(MLPProb)

    DTreeProb = DTreeProb.reshape(4, 1)
    #print(DTreeProb)
    MLPProb = MLPProb.reshape(4, 1)
    #print(MLPProb)

    a = np.hstack((DTreeProb, MLPProb))
    df = pd.DataFrame(a, columns=['DecisionTree', 'MLPClassifier'], index=classes)
    print(df)
    #plotTable(a) #visualization table

def plotTable(a):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(a, columns=['Decision Tree','MLP Classifier'], index=['0','1','2','3'])
    print(df)
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()
    plt.show()


class ClassAverageAggregate:
    def __init__(self, classValue):
        self.classValue = classValue
        self.num = 0
        self.aggregate = 0
        self.averageAggregate = 0

    def addOne(self,prob):
        self.num +=1
        self.aggregate += prob
        pass

    def calculate(self):
        if self.num != 0:
            self.averageAggregate = self.aggregate/self.num
        return self.averageAggregate

def getAverageAggregate(probAll):
    class0 = ClassAverageAggregate(0) #d
    class1 = ClassAverageAggregate(1) #h
    class2 = ClassAverageAggregate(2) #o
    class3 = ClassAverageAggregate(3) #s

    #print(len(probAll))
    for i in probAll:
        i = list(i)
        prob = max(i)
        classId = i.index(prob)
        #print(prob, i,classId)
        if classId == 0:
            class0.addOne(prob)
        elif classId == 1:
            class1.addOne(prob)
        elif classId == 2:
            class2.addOne(prob)
        elif classId == 3:
            class3.addOne(prob)

    average = []
    average.append(class0.calculate())
    average.append(class1.calculate())
    average.append(class2.calculate())
    average.append(class3.calculate())
    return average

def predicteProbStatistic(DTreeProb, MLPProb, classes):
    print('*' * 10, predicteProbStatistic.__name__,'averageAggregate in 2 classifiers')
    #print('DTree.shape = ', DTreeProb.shape)
    #print('MLPProb.shape = ', MLPProb.shape)

    probAverageDTree = getAverageAggregate(DTreeProb)
    probAverageMLP = getAverageAggregate(MLPProb)
    print(classes)
    print("probAverageDTree = ", probAverageDTree)
    print("probAverageMLP = ", probAverageMLP)
    return probAverageDTree, probAverageMLP

def train(data):
    print("*"*50,'train start','*'*30)
    nRow, nCol = data.shape
    predictors = data.iloc[:, 1:]
    target = data.iloc[:, 0]

    print('predictors.shape=', predictors.shape)
    print('target.shape=', target.shape)
    #print(predictors.iloc[0, 0])
    #print(target.iloc[:3])

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    print('pred_train.shape = ', pred_train.shape)
    print('tar_train.shape = ', tar_train.shape)
    print('pred_test.shape', pred_test.shape)
    print('tar_test.shape', tar_test.shape)

    Dtree_Clf,trainProbDTree = DecisionTree(pred_train, pred_test, tar_train, tar_test)
    MLP_Clf, trainProbMLP = MLPClassifierModel(pred_train, pred_test, tar_train, tar_test)

    #MultinomialNB_Model(pred_train, pred_test, tar_train, tar_test)
    #KNN_classifier(pred_train, pred_test, tar_train, tar_test)
    #GaussianMode(pred_train, pred_test, tar_train, tar_test)

    plotTrainProbForClassifier(trainProbDTree[0],trainProbMLP[0], Dtree_Clf.classes_)       #Part B-Q1.2
    probAverageDTree, probAverageMLP = \
        predicteProbStatistic(trainProbDTree,trainProbMLP, Dtree_Clf.classes_)              #Part B-Q1.4

    condiProbPredicts = predictByTwoClfConditProb(trainProbDTree, trainProbMLP, probAverageDTree, probAverageMLP, Dtree_Clf.classes_) #Part B-Q1.5
    #print('condiProbPredicts = ',condiProbPredicts.shape, len(condiProbPredicts))
    print('condiProbPredicts len = ', len(condiProbPredicts))
    print('tar_test.shape = ', tar_test.shape)
    #print(tar_test[:5])
    #print(type(tar_test[0]))
    #print(tar_test[0])
    #print(condiProbPredicts[:5])
    confusionMatrix(Dtree_Clf,condiProbPredicts,tar_test)
    pass
'''
def main():
    rawdata = getData()
    train(rawdata)


if __name__ == "__main__":
    main()
