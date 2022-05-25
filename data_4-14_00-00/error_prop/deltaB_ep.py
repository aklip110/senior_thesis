#use this file for the 4-14 data
#Goal: compute delta Bx, By, Bz, their averages, |B|, and its averages with correct error propagation 

import numpy as np
import matplotlib.pyplot as plt

print("                                                 ")
print("-------------------------------------------------")

#import off data
#X-Y
fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/error_prop/off_rot.txt"
datOff = np.loadtxt(fileOff)
#Z
fileOffZ = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/error_prop/off_rot-Z.txt"
datOffZ = np.loadtxt(fileOffZ)

#import on data
#X-Y
fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/error_prop/on_rot.txt"
datOn = np.loadtxt(fileOn)
#Z
fileOnZ = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/error_prop/on_rot-Z.txt"
datOnZ = np.loadtxt(fileOnZ)

#define values------------------------------------------------
plusX = 4050
minX = -3950
plusY = 8050
minY = 50

V1 = -58654

T0 = -100 #need to verify

#define functions----------------------------------------------

def subtract(ONdata, OFFdata):
    """
    this finction takes two same-shape arrasy as input
    it subtracts Bx, By, Bz (on - off) and computes the new errors by adding in quadrature.
    RETURNS: one array that's the same shape as both initial arrays. output is non-avgd, subtracted array that has correctly propagated errors!
    """
    numRows = ONdata.shape[0]
    numCols = ONdata.shape[1]
    newDat = ONdata.copy()
    for i in range(numRows):
        #Bx, By, Bz
        newDat[i, 3] = ONdata[i, 3] - OFFdata[i, 3]
        newDat[i, 5] = ONdata[i, 5] - OFFdata[i, 5]
        newDat[i, 5] = ONdata[i, 5] - OFFdata[i, 5]
        #errors
        newDat[i, 4] = np.sqrt(ONdata[i, 4]**2 + OFFdata[i, 4]**2)
        newDat[i, 6] = np.sqrt(ONdata[i, 6]**2 + OFFdata[i, 6]**2)
        newDat[i, 8] = np.sqrt(ONdata[i, 8]**2 + OFFdata[i, 8]**2)
        #SA probe
        newDat[i, 10] = ONdata[i, 10] - OFFdata[i, 10]
        newDat[i, 11] = np.sqrt(ONdata[i, 11]**2 + OFFdata[i, 11]**2)
    return newDat
    
def WAV(data, errors):
    """
    computes a weighted average of a vector of values (data) with corresponding uncertainties (errors).
    RETURNS: x_wav: a number, RMS: a number, sd_wav: a number, x_av: number
    I am curious to compare the sd_wav and the RMS, and x_wav and x_av
    
    """
    weights = np.reciprocal(np.multiply(errors, errors))
    x_wav = (1 / np.sum(weights)) * np.sum(np.multiply(data, weights))
    x_av = np.mean(data)
    sd_wav = 1 / np.sqrt(np.sum(weights))
    RMS = np.std(data, ddof=1) / np.sqrt(len(data))
    return np.array([x_wav, sd_wav, x_av, RMS])
    
def allAvgs(array):
    """
    goal: input a nx13 element array and apply the WAV function to each of Bx, By, Bz, SA.
    output: a 1 x 13+6 element array (droppping the on/off row and the time row bc meaningless) that will get stacked onto the avgd_rot_dat array.
    output array = [T, R, V, Bx_wav, Bx_wav_sd, Bx_av, Bx_sd,...]
    """
    outputArr = np.zeros(19)
    outputArr[0:3] = array[0, 0:3]
    outputArr[3:7] = WAV(array[:, 3], array[:, 4])
    outputArr[7:11] = WAV(array[:, 5], array[:, 6])
    outputArr[11:15] = WAV(array[:, 7], array[:, 8])
    outputArr[15:19] = WAV(array[:, 10], array[:, 11])
    return outputArr
    
def getMags(data):
    """
    INPUT: data is an n x 13 array of subtracted data--from the subtract function.
    this function computes |B| from Bx, By, Bz. also computes the new errors via the formula for errors of a multivariate function.
    OUTPUT: n x 5 array: [T, R, V, |B|, |B|error]
    """
    numPts = data.shape[0]
    outDat = np.zeros(5)
    for i in range(numPts):
        # compute |B|
        Bx = data[i, 3]
        By = data[i, 5]
        Bz = data[i, 7]
        dBx = data[i, 4]
        dBy = data[i, 6]
        dBz = data[i, 8]
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        Berr = (1 / Bmag) * np.sqrt((Bx * dBx)**2 + (By * dBy)**2 + (Bz * dBz)**2)
        newRow = np.array([data[i, 0], data[i, 1], data[i, 2], Bmag, Berr])
        outDat = np.vstack([outDat, newRow])
    outDat = outDat[1:, :]
    return outDat
 
#get subtracted "delta" data------------------------------------
xyDat = subtract(datOn, datOff)
zDat = subtract(datOnZ, datOffZ)
length = len(xyDat)
Zlength = len(zDat)

#compute magnitudes |B|
xyMags = getMags(xyDat)
zMags = getMags(zDat)
print("xyMags shape: ", xyMags.shape)
print("zMags shape: ", zMags.shape)

#stack the |B| and |B|error columns onto the xyDat and zDat arrays
newcol1 = np.array([xyMags[:, 3]])
newcol1 = newcol1.T
newcol2 = np.array([xyMags[:, 4]])
newcol2 = newcol2.T
xyDat = np.hstack((xyDat, newcol1, newcol2))
print("xyDat shape after: ", xyDat.shape)

newcol3 = np.array([zMags[:, 3]])
newcol3 = newcol3.T
newcol4 = np.array([zMags[:, 4]])
newcol4 = newcol4.T
zDat = np.hstack((zDat, newcol3, newcol4))
print("zDat shape after: ", zDat.shape)

#now the xyDat and zDat arrasy have 15 columns--update following code


#average XY data to get <Bx>, <By>, <Bz> vs. x and y----------------------------------------
T = xyDat[:, 0] #1st column
R = xyDat[:, 1] #2nd column
V = xyDat[:, 2] #3rd column

print("T values: ", np.unique(T))
print("R values: ",np.unique(R))
print("V values: ",np.unique(V))
numT = len(np.unique(T))
numR = len(np.unique(R))
numV = len(np.unique(V))

numPoints = numT * numR * numV
print("total pts in xy sweep: ", numPoints)
#data arrays
avgd_rot_dat = np.zeros(13 + 6 + 4)
rot_dat = np.zeros(13 + 2)
#there is no "non-time-corrected" data in this case

for t in np.unique(T):
    for r in np.unique(R):
        for v in np.unique(V):
            vals_to_avg = np.zeros(13 + 2)
            for i in range(length):
                if (xyDat[i, 0] == t) and (xyDat[i, 1] == r) and (xyDat[i, 2] == v):
                    #note: data has already been rotated
                    #print("YES")
                    newRow = xyDat[i, :]
                    #add rotated row to vals_to_avg and rot_dat (no time-correction)
                    vals_to_avg = np.vstack([vals_to_avg, newRow])
                    rot_dat = np.vstack([rot_dat, newRow])
            #grab only nonzero rows
            vals_to_avg = vals_to_avg[1:, :]
            #use the allAvgs function for each of Bx, By, Bz, SA (will be adding two new columns per probe
            magAvgs = WAV(vals_to_avg[:, 13], vals_to_avg[:, 14])
            avgd_rot_dat = np.vstack([avgd_rot_dat, np.hstack((allAvgs(vals_to_avg), magAvgs))])
            
avgd_rot_dat = avgd_rot_dat[1:, :]
rot_dat = rot_dat[1:, :]

np.savetxt("averaged_ep.txt", avgd_rot_dat, delimiter=" ")
np.savetxt("ep.txt", rot_dat, delimiter=" ")

#double check the shapes of the arrays
print("avgd_rot_dat array shape: ", avgd_rot_dat.shape)
print("rot_dat array shape: ", rot_dat.shape)

#average Z data to get <Bx>, <By>, <Bz> vs. z-----------------------------------------------
print("-----")
Z = zDat[:, 2] #3rd column
numZ = np.unique(Z)
print("numZ: ", numZ)

avgd_rot_Zdat = np.zeros(13 + 6 + 4)
rot_Zdat = np.zeros(13 + 2)

for zval in numZ:
    #print("zval: ", zval)
    zvals_to_avg = np.zeros(13 + 2)
    for i in range(Zlength):
        #print("dataval: ", Zdat[i, 2])
        if zDat[i, 2] == zval:
            #row is already transformed
            newZRow = zDat[i, :]
            zvals_to_avg = np.vstack([zvals_to_avg, newZRow])
            rot_Zdat = np.vstack([rot_Zdat, newZRow])
    #grab non-zero rows
    zvals_to_avg = zvals_to_avg[1:, :]
    #use allAvgs function
    zmagAvgs = WAV(zvals_to_avg[:, 13], zvals_to_avg[:, 14])
    avgd_rot_Zdat = np.vstack([avgd_rot_Zdat, np.hstack((allAvgs(zvals_to_avg), zmagAvgs))])

avgd_rot_Zdat = avgd_rot_Zdat[1:, :]
rot_Zdat = rot_Zdat[1:, :]

np.savetxt("averaged_ep-Z.txt", avgd_rot_Zdat, delimiter=" ")
np.savetxt("ep-Z.txt", rot_Zdat, delimiter=" ")

#double check the shapes of the arrays
print("avgd_rot_Zdat array shape: ", avgd_rot_Zdat.shape)
print("rot_Zdat array shape: ", rot_Zdat.shape)

#-----------------------------------------------------------
#plots vs. x
#V= -58654

xRight = np.zeros(9)
xLeft = np.zeros(9)

for i in range(len(avgd_rot_dat)):
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == plusX):
        x = (22.7 / 30000) * (avgd_rot_dat[i,0] - T0)
        addRow = [x, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
        #print(addRow)
        xRight = np.vstack([xRight, addRow])

    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == minX):
        x = (22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))
        addRow = [x, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
        #print(addRow)
        xLeft = np.vstack([xLeft, addRow])
        
xRight = xRight[1:, :]
xLeft = xLeft[1:, :]

# <Bx> vs. x
plt.scatter(xRight[:, 0], xRight[:, 1],  color="red", label="R = "+str(plusX))
plt.scatter(xLeft[:, 0], xLeft[:, 1], color="maroon", label="R = "+str(minX))

plt.errorbar(xRight[:, 0], xRight[:, 1], yerr=xRight[:, 2],fmt="o", color="red")
plt.errorbar(xLeft[:, 0], xLeft[:, 1], yerr=xLeft[:, 2], fmt="o",color="maroon")

plt.xlabel("x (cm)")
plt.ylabel("<Bx> (mG)")
plt.title("<Bx> (along B0) vs x: difference")
plt.legend(loc="upper center")
plt.savefig("Bx_x-deltaB.png", dpi=500)
plt.show()

# <By> vs. x
plt.scatter(xRight[:, 0], xRight[:, 3],  color="red", label="R = "+str(plusX))
plt.scatter(xLeft[:, 0], xLeft[:, 3], color="maroon", label="R = "+str(minX))

plt.errorbar(xRight[:, 0], xRight[:, 3], yerr=xRight[:, 4],fmt="o", color="red")
plt.errorbar(xLeft[:, 0], xLeft[:, 3], yerr=xLeft[:, 4], fmt="o",color="maroon")

plt.xlabel("x (cm)")
plt.ylabel("<By> (mG)")
plt.title("<By> vs x: difference")
plt.legend(loc="upper left")
plt.savefig("By_x-deltaB.png", dpi=500)
plt.show()

# <Bz> vs. x
plt.scatter(xRight[:, 0], xRight[:, 5],  color="red", label="R = "+str(plusX))
plt.scatter(xLeft[:, 0], xLeft[:, 5], color="maroon", label="R = "+str(minX))

plt.errorbar(xRight[:, 0], xRight[:, 5], yerr=xRight[:, 6],fmt="o", color="red")
plt.errorbar(xLeft[:, 0], xLeft[:, 5], yerr=xLeft[:, 6], fmt="o",color="maroon")

plt.xlabel("x (cm)")
plt.ylabel("<Bz> (mG)")
plt.title("<Bz> (vertical) vs x: difference")
plt.legend(loc="upper center")
plt.savefig("Bz_x-deltaB.png", dpi=500)
plt.show()

#<|B|> vs. x
plt.scatter(xRight[:, 0], xRight[:, 7],  color="red", label="R = "+str(plusX))
plt.scatter(xLeft[:, 0], xLeft[:, 7], color="maroon", label="R = "+str(minX))

plt.errorbar(xRight[:, 0], xRight[:, 7], yerr=xRight[:, 8],fmt="o", color="red")
plt.errorbar(xLeft[:, 0], xLeft[:, 7], yerr=xLeft[:, 8], fmt="o",color="maroon")

plt.xlabel("x (cm)")
plt.ylabel("<|B|> (mG)")
plt.title("<|B|> vs x: difference")
plt.legend(loc="upper center")
plt.savefig("B_x-deltaB.png", dpi=500)
plt.show()

#-----------------------------------------------------------
#plots vs. y
#V= -58654

yRight = np.zeros(9)
yLeft = np.zeros(9)

for i in range(len(avgd_rot_dat)):
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == plusY):
        x = (22.7 / 30000) * (avgd_rot_dat[i,0] - T0)
        addRow = [x, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
        #print(addRow)
        yRight = np.vstack([yRight, addRow])

    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == minY):
        x = (22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))
        addRow = [x, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
        #print(addRow)
        yLeft = np.vstack([yLeft, addRow])
        
yRight = yRight[1:, :]
yLeft = yLeft[1:, :]

# <Bx> vs. y
plt.scatter(yRight[:, 0], yRight[:, 1],  color="red", label="R = "+str(plusX))
plt.scatter(yLeft[:, 0], yLeft[:, 1], color="maroon", label="R = "+str(minX))

plt.errorbar(yRight[:, 0], yRight[:, 1], yerr=yRight[:, 2],fmt="o", color="red")
plt.errorbar(yLeft[:, 0], yLeft[:, 1], yerr=yLeft[:, 2], fmt="o",color="maroon")

plt.xlabel("y (cm)")
plt.ylabel("<Bx> (mG)")
plt.title("<Bx> (along B0) vs y: difference")
plt.legend(loc="upper center")
plt.savefig("Bx_y-deltaB.png", dpi=500)
plt.show()

# <By> vs. y
plt.scatter(yRight[:, 0], yRight[:, 3],  color="red", label="R = "+str(plusX))
plt.scatter(yLeft[:, 0], yLeft[:, 3], color="maroon", label="R = "+str(minX))

plt.errorbar(yRight[:, 0], yRight[:, 3], yerr=yRight[:, 4],fmt="o", color="red")
plt.errorbar(yLeft[:, 0], yLeft[:, 3], yerr=yLeft[:, 4], fmt="o",color="maroon")

plt.xlabel("y (cm)")
plt.ylabel("<By> (mG)")
plt.title("<By> vs y: difference")
plt.legend(loc="upper left")
plt.savefig("By_y-deltaB.png", dpi=500)
plt.show()

# <Bz> vs. y
plt.scatter(yRight[:, 0], yRight[:, 5],  color="red", label="R = "+str(plusX))
plt.scatter(yLeft[:, 0], yLeft[:, 5], color="maroon", label="R = "+str(minX))

plt.errorbar(yRight[:, 0], yRight[:, 5], yerr=yRight[:, 6],fmt="o", color="red")
plt.errorbar(yLeft[:, 0], yLeft[:, 5], yerr=yLeft[:, 6], fmt="o",color="maroon")

plt.xlabel("y (cm)")
plt.ylabel("<Bz> (mG)")
plt.title("<Bz> (vertical) vs y: difference")
plt.legend(loc="upper right")
plt.savefig("Bz_y-deltaB.png", dpi=500)
plt.show()

# <|B|> vs. y
plt.scatter(yRight[:, 0], yRight[:, 7],  color="red", label="R = "+str(plusX))
plt.scatter(yLeft[:, 0], yLeft[:, 7], color="maroon", label="R = "+str(minX))

plt.errorbar(yRight[:, 0], yRight[:, 7], yerr=yRight[:, 8],fmt="o", color="red")
plt.errorbar(yLeft[:, 0], yLeft[:, 7], yerr=yLeft[:, 8], fmt="o",color="maroon")

plt.xlabel("y (cm)")
plt.ylabel("<|B|> (mG)")
plt.title("<|B|> vs y: difference")
plt.legend(loc="upper right")
plt.savefig("B_y-deltaB.png", dpi=500)
plt.show()
                                            

#-----------------------------------------------------------
#plots vs. z
#V= -58654

zvals = np.zeros(9)

for i in range(len(avgd_rot_Zdat)):
    z = (15.2 / 50000) * (avgd_rot_Zdat[i,2] - V1)
    addRow = [z, avgd_rot_Zdat[i, 5], avgd_rot_Zdat[i, 6],avgd_rot_Zdat[i, 9],avgd_rot_Zdat[i, 10],avgd_rot_Zdat[i, 13],avgd_rot_Zdat[i, 14],avgd_rot_Zdat[i, -2],avgd_rot_Zdat[i, -1]]
    #print(addRow)
    zvals = np.vstack([zvals, addRow])
    
zvals = zvals[1:, :]

# <Bx> vs. z
plt.scatter(zvals[:, 0], zvals[:, 1],  color="maroon", label="T = 0, R = 4050")

plt.errorbar(zvals[:, 0], zvals[:, 1], yerr=zvals[:, 2] ,fmt="o", color="maroon")

plt.xlabel("z (cm)")
plt.ylabel("<Bx> (mG)")
plt.title("<Bx> (along B0) vs z: difference")
plt.legend(loc="upper center")
plt.savefig("Bx_z-deltaB.png", dpi=500)
plt.show()

# <By> vs. z
plt.scatter(zvals[:, 0], zvals[:, 3],  color="maroon", label="T = 0, R = 4050")

plt.errorbar(zvals[:, 0], zvals[:, 3], yerr=zvals[:, 4] ,fmt="o", color="maroon")

plt.xlabel("z (cm)")
plt.ylabel("<By> (mG)")
plt.title("<By> vs z: difference")
plt.legend(loc="upper center")
plt.savefig("By_z-deltaB.png", dpi=500)
plt.show()

# <Bz> vs. z
plt.scatter(zvals[:, 0], zvals[:, 5],  color="maroon", label="T = 0, R = 4050")

plt.errorbar(zvals[:, 0], zvals[:, 5], yerr=zvals[:, 6] ,fmt="o", color="maroon")

plt.xlabel("z (cm)")
plt.ylabel("<Bz> (mG)")
plt.title("<Bz> (vertical) vs z: difference")
plt.legend(loc="upper center")
plt.savefig("Bz_z-deltaB.png", dpi=500)
plt.show()

# <|B|> vs. z
plt.scatter(zvals[:, 0], zvals[:, 7],  color="maroon", label="T = 0, R = 4050")

plt.errorbar(zvals[:, 0], zvals[:, 7], yerr=zvals[:, 8] ,fmt="o", color="maroon")

plt.xlabel("z (cm)")
plt.ylabel("<|B|> (mG)")
plt.title("<|B|> vs z: difference")
plt.legend(loc="upper center")
plt.savefig("B_z-deltaB.png", dpi=500)
plt.show()

print("-------------------------------------------------")
print("                                                 ")
