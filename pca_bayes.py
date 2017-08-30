# TODO:
# update slides, methods, pretty-fy the presentation

import os, struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import pandas as pd
import scipy.sparse as sparse
import scipy.linalg as linalg

# CONSTANTS
TRAINING_DATA_FILE = 'X_a_With_Class_Label.xlsx'
TESTING_DATA_FILE = 'X_a_With_Class_Label_only_2016.xlsx'

def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

def readExcelRange(excelfile, sheetname="Sheet1", startrow=1, endrow=1, startcol=1, endcol=1):
    from pandas import read_excel
    values = (read_excel(excelfile, sheetname, header=None)).values
    return values[startrow-1:endrow, startcol-1:endcol]

def readExcel(excelfile, **args):
    if args:
        data = readExcelRange(excelfile, **args)
    else:
        data = readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0] == 1:
        return data[0]
    else:
        return data

def writeExcelData(x, excelfile, sheetname, startrow, startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df = DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname, startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()

def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names

def Make2DHistogramClassifier(X1,X2,T,B,xmin,xmax,ymin,ymax):
    HP=np.zeros((B,B)).astype(float)
    HN=np.zeros((B,B)).astype(float)
    binindices_x=(np.round(((B-1)*(X1-xmin)/(xmax-xmin)))).astype('int32')
    binindices_y=(np.round(((B-1)*(X2-ymin)/(ymax-ymin)))).astype('int32')
    for i,b1 in enumerate(binindices_x):
        b1 = max(b1,0)
        b1 = min(b1,24)
        by = max(binindices_y[i],0)
        by = min(by,24)
        if T[i]==1:
            HP[b1][by]+= 1;
        else:
            HN[b1][by]+= 1;
    return [HP,HN]

def queryHisto(histoData, queryX, queryY, xmin, xmax, ymin, ymax):
    bin_p1=(np.round(((24)*(queryX-xmin)/(xmax-xmin)))).astype('int32')
    bin_p1=max(bin_p1,0)
    bin_p1=min(24,bin_p1)
    bin_p2=(np.round(((24)*(queryY-ymin)/(ymax-ymin)))).astype('int32')
    bin_p2=max(bin_p2,0)
    bin_p2=min(24,bin_p2)
    try:
        return histoData[0][bin_p1][bin_p2]/float(histoData[0][bin_p1][bin_p2]+histoData[1][bin_p1][bin_p2])
    except ZeroDivisionError:
        return -1

def scatterplot(pos, neg, show_graph):
    plt.scatter(pos[:,0],pos[:,1], color='r', alpha=.3, s= 20)
    plt.scatter(neg[:,0],neg[:,1], color='b', alpha=.3, s = 20)
    if show_graph:
        show()

def removeNan(data):
    return data[~pd.isnull(data).any(axis=1)];

def getPrinComp(X):
    mu = np.mean(X, axis=0);
    Z = X - mu;
    C = np.cov(Z.astype(float), rowvar=False)
    [lmd, V] = np.linalg.eigh(C)
    lmd = np.flipud(lmd)
    V = np.flipud(V.T)
    P = np.dot(Z, V.T);
    R = np.dot(P,V);
    # print ((np.dot(P[:,0:2],V[0:2,:])+mu)[0])
    # print (X[0])
    return {"P":P, "mu":mu, "V":V, "lmd":lmd}

def getAccuracy(testset, histo, labels, xmin, xmax, ymin, ymax):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for i,b in enumerate(testset):
        probPos = queryHisto(histo, b[0], b[1], np.min(testset[:,0]), np.max(testset[:,0]), np.min(testset[:,1]), np.max(testset[:,1]))
        if probPos < 0.5:
            if labels[i] == -1:
                trueNeg += 1
            else:
                falseNeg += 1
        else:
            if labels[i] == 1:
                truePos += 1
            else:
                falseNeg += 1
    print "******Histogram Confusion Matrix******"
    print np.array([[truePos,falseNeg],[falsePos,trueNeg]])
    print "Accuracy:", (truePos+trueNeg)/float(truePos+trueNeg+falsePos+falseNeg)
    print "Sensitivity:", (truePos)/float(truePos+falseNeg)
    print "Specificity:", (trueNeg)/float(falsePos+trueNeg)
    print "PPV:", (truePos)/float(truePos+falsePos)
    print "NPV:", (trueNeg)/float(trueNeg+falseNeg)

def density(target, data):
    cv = np.cov(data, rowvar=False)
    dt = np.linalg.det(cv)
    a = np.linalg.inv(cv)
    tg = target[0]-np.mean(data[0]), target[1]-np.mean(data[1])
    return (1/(2*math.pi*pow(dt,0.5)))*math.exp(-.5*(a[0][0]*pow(tg[0],2)+2*(a[0][1]*tg[0]*tg[1])+a[1][1]*pow(tg[1],2)))

def calcWeights(data, labels):
    return np.dot(np.linalg.pinv(data), labels)

def linear(data, labels):
    W = calcWeights(data, labels);
    linear_tPos = 0
    linear_fPos = 0
    linear_fNeg = 0
    linear_tNeg = 0
    for i,b in enumerate(data):
        predict = np.sign(np.dot(b, W))
        if predict == 1:
            if labels[i] == 1:
                linear_tPos += 1
            else:
                linear_fPos += 1
        else:
            if labels[i] == 1:
                linear_fNeg += 1
            else:
                linear_tNeg += 1
    print "******Linear Confusion Matrix******"
    print np.array([[linear_tPos,linear_fNeg],[linear_fPos,linear_tNeg]])
    print "Accuracy:", (linear_tPos+linear_tNeg)/float(linear_tPos+linear_tNeg+linear_fPos+linear_fNeg)
    print "Sensitivity:", (linear_tPos)/float(linear_tPos+linear_fNeg)
    print "Specificity:", (linear_tNeg)/float(linear_fPos+linear_tNeg)
    print "PPV:", (linear_tPos)/float(linear_tPos+linear_fPos)
    print "NPV:", (linear_tNeg)/float(linear_tNeg+linear_fNeg)
    return W

def exportText(filename, data, labels):
    p_out = open(filename, "w")
    for i,b in enumerate(data):
        p_out.write(str(b[0])+" "+str(b[1])+" "+str(labels[i])+"\n");
    p_out.close()

def generateBayesian(P, P_pos, P_neg, labels):
    bayes_tPos = 0
    bayes_fNeg = 0
    bayes_fPos = 0
    bayes_tNeg = 0
    for i,b in enumerate(P):
        try:
            correct = False
            dPos = density(b[0:2],P_pos[:,:2])
            dNeg = density(b[0:2],P_neg[:,:2])
            probPos = dPos / float(dPos+dNeg)
            if probPos > 0.5:
                if labels[i] == 1:
                    bayes_tPos += 1
                    correct = True
                else:
                    bayes_fPos += 1
            else:
                if labels[i] == 1:
                    bayes_fNeg += 1
                else:
                    bayes_tNeg += 1
                    correct = True
            # print i, ":", probPos, correct
        except OverflowError:
            print i, ":", "OverflowError"
    print "******Bayesian Confusion Matrix******"
    print np.array([[bayes_tPos,bayes_fNeg],[bayes_fPos,bayes_tNeg]])
    print "Accuracy:", (bayes_tPos+bayes_tNeg)/float(bayes_tPos+bayes_tNeg+bayes_fPos+bayes_fNeg)
    print "Sensitivity:", (bayes_tPos)/float(bayes_tPos+bayes_fNeg)
    print "Specificity:", (bayes_tNeg)/float(bayes_fPos+bayes_tNeg)
    print "PPV:", (bayes_tPos)/float(bayes_tPos+bayes_fPos)
    print "NPV:", (bayes_tNeg)/float(bayes_tNeg+bayes_fNeg)

def main():
    # Load data files, strip NaN data
    data = readExcel(TRAINING_DATA_FILE, sheetname="X_a_With_Class_Label", startrow=2, endrow=1297, startcol=1, endcol=10);
    testdata = readExcel(TESTING_DATA_FILE, sheetname="X_a_With_Class_Label", startrow=2, endrow=1297, startcol=1, endcol=10);
    X = removeNan(data).astype(float);
    X_test = removeNan(testdata).astype(float);

    # Generate classifier data
    linear_weights = linear(np.c_[np.ones(X.shape[0]).T, X[:,:9]], X[:,9]);
    PCAClassifier = getPrinComp(X[:,:9])
    P = PCAClassifier["P"]
    Z_test = X_test[:,:9] - PCAClassifier["mu"]
    P_test = np.dot(Z_test, PCAClassifier["V"].T)
    exportText("P_minus2016.txt", P, X[:,9]);

    # Separate data into positive/negative classes to make Bayesian classifier
    P_pos =[]
    P_neg =[]
    for i,b in enumerate(P):
        if X[i,9] == 1:
            P_pos.append(b)
        else:
            P_neg.append(b)
    P_pos = np.array(P_pos)
    P_neg = np.array(P_neg)
    # scatterplot(P_pos,P_neg,True)

    # Customize boundaries to account for outliers in histogram
    xq1 = np.percentile(P[:,0],25)
    xq3 = np.percentile(P[:,0],75)
    yq1 = np.percentile(P[:,1],25)
    yq3 = np.percentile(P[:,1],75)
    x_lower = xq1 - 1.5*(xq3-xq1)
    x_upper = xq3 + 1.5*(xq3-xq1)
    y_lower = yq1 - 1.5*(yq3-yq1)
    y_upper = yq3 + 1.5*(yq3-yq1)
    histo = Make2DHistogramClassifier(P[:,0], P[:,1], X[:,9], 25, x_lower, x_upper, y_lower, y_upper)

    # Output histogram classifier accuracy metrics
    getAccuracy(P, histo, X[:,9], x_lower, x_upper, y_lower, y_upper)

    # Generate Bayesian classifier
    generateBayesian(P, P_pos, P_neg, X[:,9])


    print "******2016 data as testing set******"
    Xa_test = np.c_[np.ones(X_test.shape[0]).T, X_test[:,:9]]
    histo_null,histo_tPos,histo_fPos,histo_fNeg,histo_tNeg = 0,0,0,0,0
    bayes_tPos,bayes_fPos,bayes_fNeg,bayes_tNeg = 0,0,0,0
    linear_tPos,linear_fPos,linear_fNeg,linear_tNeg = 0,0,0,0

    for i,b in enumerate(P_test):
        histoResult = queryHisto(histo, b[0], b[1], x_lower, x_upper, y_lower, y_upper)
        dp = density(b[0:2],P_pos)
        dn = density(b[0:2],P_neg)
        bayesResult = dp/float(dp+dn)
        linearResult = np.dot(Xa_test[i],linear_weights)
        groundTruth = X_test[i,9]
        if groundTruth > 0:
            if math.isnan(histoResult):
                histo_null += 1
            elif histoResult > 0.5:
                histo_tPos += 1
            else:
                histo_fNeg += 1
            if bayesResult > 0.5:
                bayes_tPos += 1
            else:
                bayes_fNeg += 1
            if linearResult > 0:
                linear_tPos += 1
            else:
                linear_fNeg += 1
        else:
            if math.isnan(histoResult):
                histo_null += 1
            elif histoResult > 0.5:
                histo_fPos += 1
            else:
                histo_tNeg += 1
            if bayesResult > 0.5:
                bayes_fPos += 1
            else:
                bayes_tNeg += 1
            if linearResult > 0:
                linear_fPos += 1
            else:
                linear_tNeg += 1
    print "Histogram Classifier:"
    print "Unclassifiable:", histo_null
    print np.array([[histo_tPos,histo_fNeg],[histo_fPos,histo_tNeg]])
    print "Accuracy:", (histo_tPos+histo_tNeg)/float(histo_tPos+histo_tNeg+histo_fPos+histo_fNeg)
    print "Sensitivity:", (histo_tPos)/float(histo_tPos+histo_fNeg)
    print "Specificity:", (histo_tNeg)/float(histo_fPos+histo_tNeg)
    print "PPV:", (histo_tPos)/float(histo_tPos+histo_fPos)
    print "NPV:", (histo_tNeg)/float(histo_tNeg+histo_fNeg)
    print "Bayesian Classifier:"
    print np.array([[bayes_tPos,bayes_fNeg],[bayes_fPos,bayes_tNeg]])
    print "Accuracy:", (bayes_tPos+bayes_tNeg)/float(bayes_tPos+bayes_tNeg+bayes_fPos+bayes_fNeg)
    print "Sensitivity:", (bayes_tPos)/float(bayes_tPos+bayes_fNeg)
    print "Specificity:", (bayes_tNeg)/float(bayes_fPos+bayes_tNeg)
    print "PPV:", (bayes_tPos)/float(bayes_tPos+bayes_fPos)
    print "NPV:", (bayes_tNeg)/float(bayes_tNeg+bayes_fNeg)
    print "Linear Classifier:"
    print np.array([[linear_tPos,linear_fNeg],[linear_fPos,linear_tNeg]])
    print "Accuracy:", (linear_tPos+linear_tNeg)/float(linear_tPos+linear_tNeg+linear_fPos+linear_fNeg)
    print "Sensitivity:", (linear_tPos)/float(linear_tPos+linear_fNeg)
    print "Specificity:", (linear_tNeg)/float(linear_fPos+linear_tNeg)
    print "PPV:", (linear_tPos)/float(linear_tPos+linear_fPos)
    print "NPV:", (linear_tNeg)/float(linear_tNeg+linear_fNeg)
    print "******End 2016 data testing set******"

    plt.scatter(P_pos[:,0],P_pos[:,1], color='r', alpha=.3, s=20)
    plt.scatter(P_neg[:,0],P_neg[:,1], color='b', alpha=.3, s=20)
    plt.scatter(P_test[:,0],P_test[:,1],color='g',alpha=.3, s=20)
    # show()


main();
