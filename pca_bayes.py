import os, struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import pandas as pd
import scipy.sparse as sparse
import scipy.linalg as linalg

# CONSTANTS
DATA_FILE = 'X_a_With_Class_Label.xlsx'

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
        if T[i]==1:
            HP[b1][binindices_y[i]]+= 1;
        else:
            HN[b1][binindices_y[i]]+= 1;
    return [HP,HN]

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
    V = np.flipud(V)
    P = np.dot(Z, V.T);
    return P

def main():
    data = readExcel(DATA_FILE, sheetname="X_a_With_Class_Label", startrow=2, endrow=1297, startcol=1, endcol=10);
    X = removeNan(data).astype(float);
    P = getPrinComp(X[:,:9])
    scatterplot(P, X[:,9], True)
    histo = Make2DHistogramClassifier(P[:,0], P[:,1], X[:,9], 25, np.min(P[:,0]), np.max(P[:,0]), np.min(P[:,1]), np.max(P[:,1]))


main();
