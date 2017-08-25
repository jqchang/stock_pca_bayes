import os, struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import pandas as pd
import scipy.sparse as sparse
import scipy.linalg as linalg

# CONSTANTS
DATA_FILE = 'X_a_With_Class_Label_2016_new.xlsx'

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
    for i,b in enumerate(binindices_x):
        if b < 0:
            binindices_x[i] = 0
        if b > 24:
            binindices_x = 24
    binindices_y=(np.round(((B-1)*(X2-ymin)/(ymax-ymin)))).astype('int32')
    for i,b in enumerate(binindices_y):
        if b < 0:
            binindices_y[i] = 0
        if b > 24:
            binindices_y = 24
    for i,b1 in enumerate(binindices_x):
        if T[i]==1:
            HP[b1][binindices_y[i]]+= 1;
        else:
            HN[b1][binindices_y[i]]+= 1;
    return [HP,HN]

def queryHisto(histoData, queryX, queryY, xmin, xmax, ymin, ymax):
    bin_p1=(np.round(((24)*(queryX-xmin)/(xmax-xmin)))).astype('int32')
    bin_p2=(np.round(((24)*(queryY-ymin)/(ymax-ymin)))).astype('int32')
    try:
        return histoData[0][bin_p1][bin_p2]/float(histoData[0][bin_p1][bin_p2]+histoData[1][bin_p1][bin_p2])
    except ZeroDivisionError:
        return -1

def MakeBayesClassifier():
    pass

def scatterplot(P, T, show_graph):
    cols=np.zeros((alen(T), 4));
    for i in range(len(T)):
        if T[i] == -1:
            cols[i]=[1,0,0,0.25];
        elif T[i] == 1:
            cols[i]=[0,1,0,0.25];
        else:
            cols[i]=[0,0,1,0.25];
    randomorder=permutation(arange(alen(T))); #Don't worry about this stuff. Just makes a pretty picture

    fig = figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.scatter(P[randomorder,1],P[randomorder,0],s=5,linewidths=0,facecolors=cols[randomorder,:],marker="o");
    ax.set_aspect('equal');
    gca().invert_yaxis();
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
    return P

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
    # print "True Positive:", truePos
    # print "True Negative:", trueNeg
    # print "False Positive:", falsePos
    # print "False Negative:", falseNeg
    accuracy = (truePos+trueNeg)/float(truePos+trueNeg+falsePos+falseNeg)
    return accuracy

def density(target, data):
    cv = np.cov(data)
    dt = np.linalg.det(cv)
    a = np.linalg.inv(cv)
    tg = target[0]-np.mean(data[0]), target[1]-np.mean(data[1])
    return (1/(2*math.pi*pow(dt,0.5)))*math.exp(-.5*(a[0][0]*pow(tg[0],2)+2*(a[0][1]*tg[0]*tg[1])+a[1][1]*pow(tg[1],2)))

def main():
    data = readExcel(DATA_FILE, sheetname="X_a_With_Class_Label", startrow=2, endrow=1517, startcol=1, endcol=10);
    X = removeNan(data).astype(float);
    P = getPrinComp(X[:,:9])
    for i,b in enumerate(X[:,:9]):
        print i,b
        print i,P[i]
    scatterplot(P, X[:,9], True)
    histo = Make2DHistogramClassifier(P[:,0], P[:,1], X[:,9], 25, np.min(P[:,0]), np.max(P[:,0]), np.min(P[:,1]), np.max(P[:,1]))
    histoAccuracy = getAccuracy(P, histo, X[:,9], np.min(P[:,0]), np.max(P[:,0]), np.min(P[:,1]), np.max(P[:,1]))
    print "accuracy:", histoAccuracy
    # for i,b in enumerate(P):
    #     try:
    #         d = density(b[0:2],P[:,:2])
    #         print i, ":", d
    #     except OverflowError:
    #         print i, ":", "OverflowError"
    # print "Accuracy:", histoAccuracy


main();
