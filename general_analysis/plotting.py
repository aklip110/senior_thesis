import numpy as np
import matplotlib.pyplot as plt
from get_data import *
from deltaB_general import *

#plotting functions

def Plots(Right, Left, rColor, lColor, dataName, filePath,  xyz, plus=0, minus=0, legend=True):
    """
    INPUT: Right (nx9 array) of data for RHS, Left (nx9 array) of data for LHS--set both same if dont want to distinguish; r/lCOlor is right/left color; dataName (string) is data/time stamp; filePath (string) is path to senior_thesis directory; plus/minus are the plus and minus R values (only used with x or y data); xyz (string: "x", "y", or "z") specifies which axis we're sweeping; legend (Bool) is False for z.
    FUNCTION: makes 4 plots given the input data: <Bx>, <By>, <Bz>, <|B|>.
    OUTPUT: nothing, just saves plots to the dataName directory.
    """
    # <Bx> vs. x
    fig1 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0], Right[:, 1],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0], Left[:, 1], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0], Right[:, 1], yerr=Right[:, 2],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0], Left[:, 1], yerr=Left[:, 2], fmt="o",color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bx> (mG)")
    plt.title("<Bx> (along B0) vs "+ xyz +": " + dataName + "FIXED")
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + "/FIXEDplots/" + "Bx_" + xyz + "-deltaB_" + dataName + "-FIXED.png", dpi=500)
    #plt.show()
    
    # <By> vs. x
    fig2 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0], Right[:, 3],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0], Left[:, 3], color=lColor, label="R = "+str(minus))

    #plt.errorbar(Right[:, 0], Right[:, 3], yerr=Right[:, 4],fmt="o", color=rColor)
    #plt.errorbar(Left[:, 0], Left[:, 3], yerr=Left[:, 4], fmt="o",color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<By> (mG)")
    plt.title("<By> vs "+ xyz +": " + dataName + "FIXED")
    if legend == True:
        plt.legend(loc="upper left")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + "/FIXEDplots/" + "By_" + xyz + "-deltaB_" + dataName + "-FIXED.png", dpi=500)
    #plt.show()
    
    # <Bz> vs. x
    fig3 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0], Right[:, 5],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0], Left[:, 5], color=lColor, label="R = "+str(minus))

    #plt.errorbar(Right[:, 0], Right[:, 5], yerr=Right[:, 6],fmt="o", color=rColor)
    #plt.errorbar(Left[:, 0], Left[:, 5], yerr=Left[:, 6], fmt="o", color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bz> (mG)")
    plt.title("<Bz> (vertical) vs "+ xyz +": " + dataName + "FIXED")
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + "/FIXEDplots/" + "Bz_" + xyz + "-deltaB_" + dataName + "-FIXED.png", dpi=500)
    #plt.show()
    
    #<|B|> vs. x
    fig4 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0], Right[:, 7],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0], Left[:, 7], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0], Right[:, 7], yerr=Right[:, 8],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0], Left[:, 7], yerr=Left[:, 8], fmt="o",color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<|B|> (mG)")
    plt.title("<|B|> vs "+ xyz +": " + dataName + "FIXED")
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + "/FIXEDplots/" + "B_" + xyz + "-deltaB_" + dataName + "-FIXED.png", dpi=500)
    #plt.show()
    
    return
    
def make_all_plots(filepath, paramArray):
    print("Parameter array: ")
    print(paramArray)
    print("Parameter array shape: ", paramArray.shape)
    for n in range(4,5):
        dataname = paramArray[n][0]
        plusX = paramArray[n][1]
        minX = paramArray[n][2]
        plusY = paramArray[n][3]
        minY = paramArray[n][4]
        V1 = paramArray[n][5]
        T0 = paramArray[n][10] #need to verify

        xRight, xLeft, yRight, yLeft, zAll = complete_data(filepath, dataname, plusX, minX, plusY, minY, V1, T0)

        Plots(xRight, xLeft, "red", "maroon", dataname, filepath, "x", plus=plusX, minus=minX)

        Plots(yRight, yLeft, "red", "maroon", dataname, filepath, "y",  plus=plusY, minus=minY)

        Plots(zAll, zAll, "maroon", "maroon", dataname, filepath, "z", legend=False)
    return
    
def import_comsol(filePath, fileName):
    """
    INPUT: filepath (string) is path to senior_thesis directory; fileName (string: "Bi(j)" | i,j = x, y, or z)
    FUNCTION: imports the specified comsol file
    OUTPUT: CSarr (nx2 array) of x and y values to plot comsol data.
    """
    file = filePath + "comsol_dat/B017fieldprofile_Metglas_noPB_" + fileName + ".txt"
    CSarr = np.loadtxt(file)
    return CSarr
    
def plot_comsol(i, j, filePath, cutoff, fileName, pArr, symmetric=False):
    """
    INPUT: i, j (strings: "x", "y" or "z") that specify Bi(j); filePath to senior_thesis directory; cutoff (float \in [0, 40 cm]) is the x/y/z value we want to plot up to, symmetric=True will plot left and right data, fileName (string) is the name of the data we will overlay. Note: this comparison is really only meaningful for the Bx data. Perhaps i'll also add functionality to plot |B|...
    FUNCTION: plot all 9 comsol datasets.
    OUTPUT: none
    """
    #import comsol data
    if i=="mag":
        dat = import_comsol(filePath, "Bx(" + j + ")")
    else:
        dat = import_comsol(filePath, "B" + i + "(" + j + ")")
    if symmetric == True:
        dat2 = np.hstack((np.array([-dat[:, 0]]).T, np.array([dat[:, 1]]).T))
        #print(dat2)
        dat = np.vstack([dat, dat2])
    #print(dat)
    counter = 0
    
    plottingDat = np.zeros(2)
    for k in range(dat.shape[0]):
        #print("k: ", k)
        if (np.abs(dat[k, 0]) <= cutoff / 100):
            #print("yes")
            #print(dat[k, 1])
            plottingDat = np.vstack([plottingDat, dat[k, :]])
            
    plottingDat = plottingDat[1:, :]
    plottingDat = plottingDat[plottingDat[:, 0].argsort(), :]
    #a = a[a[:, 0].argsort()]
    
    #import actual data for comparison
    colVal = 0
    if i=="x":
        colVal = 1
    if i=="y":
        colVal = 3
    if i=="z":
        colVal = 5
    if i=="mag":
        colVal = 7
    #first grab the parameter values from the parameter file
    for k in range(pArr.shape[0]):
        if pArr[k][0] == fileName:
            print("yes")
            xRight, xLeft, yRight, yLeft, zAll = complete_data(filePath, fileName, pArr[k][1], pArr[k][2], pArr[k][3], pArr[k][4], pArr[k][5], pArr[k][10])
            if j=="x":
                if i!= "mag":
                    xdat = np.hstack([xRight[:, 0], xLeft[:, 0]])
                    ydat = np.hstack([xRight[:, colVal], xLeft[:, colVal]])
                    errs = np.hstack([xRight[:, colVal + 1], xLeft[:, colVal + 1]])
                if i== "mag":
                    X = np.hstack([xRight[:, 1], xLeft[:, 1]])
                    Y = np.hstack([xRight[:, 3], xLeft[:, 3]])
                    Z = np.hstack([xRight[:, 5], xLeft[:, 5]])
                    ydat = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))
                    xdat = np.hstack([xRight[:, 0], xLeft[:, 0]])
                #rescale the comsol data using minima
                minDat = np.min(ydat)
                minCS = np.min(plottingDat[:, 1])
                ydatCS = plottingDat[:, 1] * (minDat / minCS)
                xdatCS = plottingDat[:, 0] * 100
            if j=="y":
                xdat = np.hstack([yRight[:, 0], yLeft[:, 0]])
                ydat = np.hstack([yRight[:, colVal], yLeft[:, colVal]])
                errs = np.hstack([yRight[:, colVal + 1], yLeft[:, colVal + 1]])
                #rescale the comsol data using minima
                minDat = np.min(ydat)
                minCS = np.min(plottingDat[:, 1])
                ydatCS = plottingDat[:, 1] * (minDat / minCS)
                xdatCS = plottingDat[:, 0] * 100
            if j=="z":
                xdat = zAll[:, 0]
                ydat = zAll[:, colVal]
                errs = zAll[:, colVal + 1]
                #rescale the comsol data using minima
                maxDat = np.max(ydat)
                maxCS = np.max(plottingDat[:, 1])
                ydatCS = plottingDat[:, 1] * (maxDat / maxCS)
                xdatCS = plottingDat[:, 0] * 100
    fig = plt.figure()
    plt.ion()
    plt.scatter(xdat, ydat,  color="maroon")
    #plt.errorbar(xdat, ydat, yerr=errs, fmt="o", color="maroon")
    
    plt.scatter(xdatCS, ydatCS, color="gray")
    plt.plot(xdatCS, ydatCS, color="gray")
    plt.xlabel(j + " (cm)")
    plt.ylabel("B" + i)
    plt.title("B" + i + "(" + j + ") comsol comparison: " + fileName + "FIXED")
    plt.ioff()
    plt.savefig(filePath + "comsol_dat/B"+ i + "(" + j + ")_" + fileName + "corrected-FIXED.png")
    #plt.show()
    return
    
#----------------------------

paramFile = "/Users/alexandraklipfel/Desktop/senior_thesis/general_analysis/parameters.txt"
paramArray = np.genfromtxt(paramFile, dtype=['U11', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8'])

filepath = "/Users/alexandraklipfel/Desktop/senior_thesis/"

#plot_comsol("mag", "x", filepath, 20, "4-05_21-36", paramArray, symmetric=True)
#plot_comsol("mag", "y", filepath, 20,  "4-05_21-36", paramArray, symmetric=True)
#plot_comsol("mag", "z", filepath, 20, "4-05_21-36", paramArray)

make_all_plots(filepath, paramArray)

