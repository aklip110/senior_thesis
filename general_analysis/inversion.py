#this file contains the functions to use the optimal parameters to "fix" an individual dataset
#parameters are found by running the original data through the optimize.py script

import numpy as np
from optimize import *
from plotParams import *

def matMultErrs(mat, vec):
    """
    INPUT: mat is a 3x3 matrix and vec is a 1x3 vector of ERROR vals
    OUTPUT: a vector of new errors that will correspond to the transformed b vector with initial errors "vec"
    """
    newErrs = np.zeros(3)
    #print("------in err func------")
    for row in range(3):
        #print("row: ", row)
        #element-wise multiply row and error vec
        scaledErrs = np.multiply(np.abs(mat[row, :]), vec)
        #print("scaled Errors: ", scaledErrs)
        newErrs[row] = np.sqrt(np.dot(scaledErrs, scaledErrs))
        #print("new errors: ", newErrs)
    #print("------")
    return newErrs

def invert(XYdata, rowVal, paramArray, Xp, Xm, Yp, Ym):
    """
    num is in range(13) indicates which file, rowVal is the row index of the point we want to "fix"
    OUTPUT: fixed B vector (cc) and original field vector (bObs), vector of optimal parameters (params)
    """
    
    # import the dataset
    #dataName = paramArray[num][0]
    #XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    #XYdata = np.loadtxt(XYfile)
    #print("File: ", dataName)
    
    ######OPTIMIZATION
    x0 = [0.017, 0.017,  0.017]
    #res = minimize(bFuncSqrs, x0, args=(XYdata[rowVal, 0], XYdata[rowVal, 1], XYdata, Xp, Xm, Yp, Ym), method='Nelder-Mead', tol=1e-6 )
    res = minimize(bFuncSqrs, x0, args=(XYdata[rowVal, 0], XYdata[rowVal, 1], XYdata, Xp, Xm, Yp, Ym), method='Nelder-Mead', tol=1e-6 )
    #print("res.x: ", res.x)
    params = res.x
    ##################
    
    #print("XY vals: ", [Xp, Xm, Yp, Ym])

    #create copy of original dataset
    #fixedDat = np.copy(XYdata)
    
    #for row in [rowVal]: #range(len(XYdat)):
        #compute the rotation matrices (use function from optimize) and their inverses
        #noting that the params used depend on R
        #print(XYdata[row, :])
    r = XYdata[rowVal, 1]
    t = XYdata[rowVal, 0]
        #if r == Xp:
            #theta = params[6]
            #b = params[2]
        #elif r == Xm:
            #theta = params[7]
            #b = params[4]
        #elif r == Yp:
            #theta = params[8]
            #b = params[3]
        #elif r == Ym:
            #theta = params[9]
            #b = params[5]
        ##########
        #p = psi(params[1], b, t)
    theta = params[2]
    p = params[1]
    phi = params[0]
        ###########
        
    #print("phi: ", params[0])
    #print("psi: ", p)
    #print("theta: ", theta)
        #print("b: ", b)
        
    R1inv = np.linalg.inv(R1mat(theta))
    R2inv = np.linalg.inv(R2mat(p))
    R3inv = np.linalg.inv(R3mat(phi))
        
        #print("Matrix inverses: ")
        #print(R1mat(theta))
        #print(R1inv)
        #print(R2mat(p))
        #print(R2inv)
        #print(R3mat(params[0]))
        #print(R3inv)
        
        # for each line in the original file (on and off), Bnew = inv(R1).inv(R2).inv(R3).Bobs
    bObs = np.array([XYdata[rowVal, 3], XYdata[rowVal, 5], XYdata[rowVal, 7]])
    bObsErrs = np.array([XYdata[rowVal, 4], XYdata[rowVal, 6], XYdata[rowVal, 8]])
    #print("bObs: ", bObs)
    #print("bObsErrs: ", bObsErrs)
        
    aa = np.matmul(R3inv, bObs)
    aaErrs = matMultErrs(R3inv, bObsErrs)
    #print("aaErrs: ", aaErrs)
    #print("a: ", aa)
    bb = np.matmul(R2inv, aa)
    bbErrs = matMultErrs(R2inv, aaErrs)
    #print("bbErrs: ", bbErrs)
    #print("b: ", bb)
    cc = np.matmul(R1inv, bb)
    ccErrs = matMultErrs(R1inv, bbErrs)
    #print("ccErrs: ", ccErrs)
    #print("cc: ", cc)
        #replace the Bx, By, Bz values in each row of the dataset copy
        #fixedDat[row, 3] = cc[0]
        #fixedDat[row, 5] = cc[1]
        #fixedDat[row, 7] = cc[2]
        
    # save to the fixed dataset as "data_X-X_X-X-3d_scan-FIXED.txt"
    #np.savetxt(filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan-FIXED.txt", fixedDat)
    return cc, bObs, params, ccErrs
    
#new function that will iterate over all rows in a file and save the output to a new txt file
def invertAll(num, paramArray, vert=False):
    """
    applies the inversion function to every line in a given dataset
    OUTPUT: saves a new dataset version appended with "FIXED", also makes and saves the parameter plots if vert = False.
    """
    # import the dataset
    dataName = paramArray[num][0]
    print(dataName)
    if vert == False:
        XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    elif vert == True:
        XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-vertical_sweep.txt"
    XYdata = np.loadtxt(XYfile)
    
    Xp = paramArray[num][1]
    Xm = paramArray[num][2]
    Yp = paramArray[num][3]
    Ym = paramArray[num][4]
    
    fixedDat = np.copy(XYdata)
    paramDat = np.zeros((len(XYdata), 5))
    
    for row in range(len(XYdata)):
        fixedVec, oldVec, paramVec, fixedErrs = invert(XYdata, row, paramArray, Xp, Xm, Yp, Ym)
        print(fixedVec)
        print(oldVec)
        #data Bx, BY, Bz values
        fixedDat[row, 3] = fixedVec[0]
        fixedDat[row, 5] = fixedVec[1]
        fixedDat[row, 7] = fixedVec[2]
        fixedDat[row, 4] = fixedErrs[0]
        fixedDat[row, 6] = fixedErrs[1]
        fixedDat[row, 8] = fixedErrs[2]
        #error values
        
        paramDat[row, :] = np.hstack((XYdata[row, 0:2], paramVec))
        
    # save to the fixed dataset as "data_X-X_X-X-3d_scan-FIXED.txt"
    if vert == False:
        np.savetxt(filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan-FIXED.txt", fixedDat)
        np.savetxt(filePath + "data_" + dataName + "/paramDat-3D_scan-new.txt", paramDat)
        #make all the parameter plots
        plotParameters(num, Xp, paramArray)
        plotParameters(num, Xm, paramArray)
        plotParameters(num, Yp, paramArray)
        plotParameters(num, Ym, paramArray)
    elif vert == True:
        np.savetxt(filePath + "data_" + dataName + "/" + "data_" + dataName + "-vertical_sweep-FIXED.txt", fixedDat)
        np.savetxt(filePath + "data_" + dataName + "/paramDat-vertical_sweep.txt", paramDat)
    # Nx5 array: [T, R, phi, psi, theta] for on and off data
    
    return
    
    

paramFile = "/Users/alexandraklipfel/Desktop/senior_thesis/general_analysis/parameters.txt"
paramArray00 = np.genfromtxt(paramFile, dtype=['U11', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8'])
filePath = "/Users/alexandraklipfel/Desktop/senior_thesis/"

# for the 4-10_20-25 dataset
#Params = [ 2.85645743e-02,  3.68384585e-02, -4.58683486e-01, -4.80155171e-01, -7.45063511e+00, -4.81032652e-03, 1.38230661e-02,  1.06047588e-01, 1.64805509e-02,  7.62058643e-02]

#x0 = [0.017, 0.017,  0.017]
#res = minimize(bFuncSqrs, x0, args=(2525, 4160, 4, paramArray), method='Nelder-Mead', tol=1e-6 )
#print(res.x)

#Params2 = [ res.x[0],  res.x[1], res.x[2], 0, 0, 0, res.x[3],  0, 0,  0]

#Params3 = res.x

#print(Params)

#print(invert(Params, 4, 2)[0])
#print(invert(Params, 4, 3)[0])
#new0, old0 = invert(4, 10)
#new1, old1 = invert(4, 11)
#print(np.subtract(new0, new1))
#print(np.subtract(old0, old1))

for index in range(1):
    print(index)
    invertAll(index, paramArray00)
    #invertAll(index, paramArray00, vert=True)
