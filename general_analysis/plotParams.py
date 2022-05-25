#just a quick script to plot the paramters vs T given a file that's generated by the inversion.py script

import numpy as np
import matplotlib.pyplot as plt


def plotParameters(num, Rval, paramArray, vert=False):
    """
    Plots how the parameter estimates change with radial distance at a given R value
    """
    filePath = "/Users/alexandraklipfel/Desktop/senior_thesis/"
    dataName = paramArray[num][0]
    if vert == False:
        pFile = filePath + "data_" + dataName + "/paramDat-3D_scan.txt"
    elif vert == True:
        pFile = filePath + "data_" + dataName + "/paramDat-vertical_sweep.txt"
    pData = np.loadtxt(pFile)
    print(pFile)
    
    distvals = np.zeros(int(len(pData) / 4))
    phivals = np.zeros(int(len(pData) / 4))
    psivals = np.zeros(int(len(pData) / 4))
    thetavals = np.zeros(int(len(pData) / 4))
    
    print("shape of pData: ", pData.shape)
    i = 0
    for j in range(len(pData)):
        if pData[j, 1] == Rval:
            print("Yes: ", pData[j, 1])
            distvals[i] = pData[j, 0] * (22.7 / 30000)
            phivals[i] = pData[j, 2]
            psivals[i] = pData[j, 3]
            thetavals[i] = pData[j, 4]
            i += 1
            
    print(distvals)
    print(phivals)
    print(psivals)
    print(thetavals)
            
    #plots
    #phi
    fig1 = plt.figure()
    plt.scatter(distvals, phivals)
    plt.xlabel("distance (cm)")
    plt.ylabel("phi parameter estimate")
    plt.title("Phi param values: R = " + str(Rval))
    plt.savefig(filePath + "data_" + dataName + "/paramPlots/" + "phi" + str(Rval) + ".png", dpi=500)
    plt.show()
    
    #psi
    fig1 = plt.figure()
    plt.scatter(distvals, psivals)
    plt.xlabel("distance (cm)")
    plt.ylabel("psi parameter estimate")
    plt.title("Psi param values: R = " + str(Rval))
    plt.savefig(filePath + "data_" + dataName + "/paramPlots/" + "psi" + str(Rval) + ".png", dpi=500)
    plt.show()
    
    #theta
    fig1 = plt.figure()
    plt.scatter(distvals, thetavals)
    plt.xlabel("distance (cm)")
    plt.ylabel("theta parameter estimate")
    plt.title("Theta param values: R = " + str(Rval))
    plt.savefig(filePath + "data_" + dataName + "/paramPlots/" + "theta" + str(Rval) + ".png", dpi=500)
    plt.show()
    
    return
    
    
    


#paramFile = "/Users/alexandraklipfel/Desktop/senior_thesis/general_analysis/parameters.txt"
#paramArray00 = np.genfromtxt(paramFile, dtype=['U11', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8'])


#plotParameters(4, 4160, paramArray00)
#plotParameters(4, 8160, paramArray00)
#plotParameters(4, 160, paramArray00)
#plotParameters(4, -3840, paramArray00)
#plotParameters(4, vert=True)
