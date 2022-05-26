#least squares estimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def B0(R, Bmag, Xp, Xm, Yp, Ym):
    if R == Xp:
        b = np.array([0,0,Bmag])
    elif R == Xm:
        b = np.array([0,0,-Bmag])
    elif R == Yp:
        b = np.array([Bmag,0,0])
    elif R == Ym:
        b = np.array([-Bmag,0,0])
    return b
    
#def psi(a, b, T):
    #out = a * (22.7 / 30000) * T + b
    #return out
    
def R1mat(theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return mat
    
def R2mat(p):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(p), -np.sin(p)],
                    [0, np.sin(p), np.cos(p)]])
    return mat
    
def R3mat(phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return mat
    
def bFunc(T, R, phi, psi, theta, XYdata, Xp, Xm, Yp, Ym):
    """
    given a point (T, R, (V) not needed) and four angle parameters, this computes the field in the transformed coordinates
    """
    #print("in bFunc-----------")
    #theta contains R info--it's determined by R
    #import file
    #filePath = "/Users/alexandraklipfel/Desktop/senior_thesis/"
    #dataName = paramArray[Num][0]
    #XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    #XYdata = np.loadtxt(XYfile)
    #Xp = paramArray[Num][1]
    #Xm = paramArray[Num][2]
    #Yp = paramArray[Num][3]
    #Ym = paramArray[Num][4]
    #compute magnitude and get b0
    bmag = 0
    counter = 0
    bObsX = 0
    bObsY = 0
    bObsZ = 0
    for i in range(len(XYdata)):
        if (XYdata[i, 0] == T) and (XYdata[i, 1] == R) and (XYdata[i, 9] == 1):
            bmag += np.sqrt(((XYdata[i, 3] - XYdata[i+1, 3])**2 + (XYdata[i, 5] - XYdata[i+1, 5])**2 + (XYdata[i, 7] - XYdata[i+1, 7])**2))
            bObsX += (XYdata[i, 3] - XYdata[i+1, 3])
            bObsY += (XYdata[i, 5] - XYdata[i+1, 5])
            bObsZ += (XYdata[i, 7] - XYdata[i+1, 7])
            counter += 1
    bmag = bmag / counter #gets average magnitude at the point (T, R, V)
    #print("Average B magnitude measured: ", bmag)
    b0 = B0(R, bmag, Xp, Xm, Yp, Ym)
    #print("b0: ", b0)
    bObsVec = np.array([bObsX / counter, bObsY / counter, bObsZ / counter])
    #print("Bobsvec: ", bObsVec)
    #make rotation matrices
    #p = psi(a, b, T) * (np.pi / 180)
    p = psi
    #print("theta: ", theta)
    #print("psi: ", p)
    #print("phi: ", phi)
    R1 = R1mat(theta)
    R2 = R2mat(p)
    R3 = R3mat(phi)
    #print(R1)
    #print(R2)
    #print(R3)
    #matrix multiplication
    aa = np.matmul(R1, b0)
    #print("a: ", aa)
    bb = np.matmul(R2, aa)
    #print("b: ", bb)
    cc = np.matmul(R3, bb)
    #print("cc: ", cc)
    bvec = np.zeros(3)
    for i in range(3):
        if np.abs(cc[i]) > 10**(-10):
            bvec[i] = cc[i]
        else:
            bvec[i] = 0
    #print("bvec: ", bvec)
    #print("bvec mag: ", np.sqrt(np.dot(bvec, bvec)))
    #print("end bFunc---------")
    return np.subtract(bvec, bObsVec)
    
def bSqrs(x, XYdata):
    """
    creates the least squares sum that needs to be minimized wrt the first seven parameters. sums over all R and T values in a given file and over the three x, y, z components of the field vector.
    """
    phi = x[0]
    a = x[1]
    b1 = x[2]
    b2 = x[3]
    b3 = x[4]
    b4 = x[5]
    tXp = x[6]
    tXm = x[7]
    tYp = x[8]
    tYm = x[9]
    #dataName = paramArray[Num][0]
    #XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    #XYdata = np.loadtxt(XYfile)
    Tvals = np.unique(XYdata[:, 0])
    Rvals = np.unique(XYdata[:, 1])
    ivals = np.array([0,1,2])
    
    #print("Tvals: ", Tvals)
    #print("Rvals: ", Rvals)
    #print("ivals: ", ivals)
    
    ######################
    #Tvals = np.array([4525])
    #Rvals = np.array([Yp])
    ######################
    
    LSQ = 0
    
    for t in Tvals:
        for r in Rvals:
            #get correct theta
            if r == Xp:
                theta = tXp
                b = b1
            elif r == Xm:
                theta = tXm
                b = b2
            elif r == Yp:
                theta = tYp
                b = b3
            elif r == Ym:
                theta = tYm
                b = b4
            for i in ivals:
                #print("deltab: ", bFunc(t, r, phi, a, b, theta)[i])
                LSQ += (bFunc(t, r, phi, a, b, theta, Num)[i])**2
                #print("LSQval: ", LSQ)
    return LSQ
    
#before optimization: debug with other three R values and T values
    #seems to work with all four R values
    
def bFuncSqrs(x, T, R, XYdata, Xp, Xm, Yp, Ym):
    """
    computes LSQ value for single point based on the three field components.
    OUTPUT set of 4 optimal parameters.
    """
    #filePath = "/Users/alexandraklipfel/Desktop/senior_thesis/"
    phi = x[0]
    #a = x[1]
    #b = x[2]
    psi = x[1]
    theta = x[2]
    #dataName = paramArray[Num][0]
    #XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    #XYdata = np.loadtxt(XYfile)
    #Xp = paramArray[Num][1]
    #Xm = paramArray[Num][2]
    #Yp = paramArray[Num][3]
    #Ym = paramArray[Num][4]
    ivals = np.array([0,1,2])

    LSQ = 0
    
    for i in ivals:
        #print("deltab: ", bFunc(t, r, phi, a, b, theta)[i])
        LSQ += (bFunc(T, R, phi, psi, theta, XYdata, Xp, Xm, Yp, Ym)[i])**2
        #print("LSQval: ", LSQ)
    return LSQ
    
    
"""
paramFile = "/Users/alexandraklipfel/Desktop/senior_thesis/general_analysis/parameters.txt"
paramArray = np.genfromtxt(paramFile, dtype=['U11', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8'])
filePath = "/Users/alexandraklipfel/Desktop/senior_thesis/"

for num in range(1,13):
    print("---------------")
    Xp = paramArray[num][1]
    Xm = paramArray[num][2]
    Yp = paramArray[num][3]
    Ym = paramArray[num][4]
    print("File: ", paramArray[num][0])
    print("XY vals: ", [Xp, Xm, Yp, Ym])

    #print(bFunc(8525, Xp, 0.017, -0.67, -0.67/12 ,0.017))
    #bounds=((-np.pi/4, np.pi/4),(-1, 1),(-1, 1),(-np.pi/4, np.pi/4),(-np.pi/4, np.pi/4),(-np.pi/4, np.pi/4),(-np.pi/4, np.pi/4))
    #print(bSqrs([0.017, -0.67, -0.67/12 ,0.017, 0.05, 0.02, 0.02]))

    #implement optimization and test with one T and R value first, then go all T, then try with varius R and varius datasets
    x0 = [0.017, -0.67/12, -0.67, -0.67, -0.67, -0.67, 0.017, 0.05, 0.02, 0.02]
    res = minimize(bSqrs, x0, args=(num), method='Nelder-Mead', tol=1e-6 )
    print(res.x)
"""
