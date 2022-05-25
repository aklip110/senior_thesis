# imports rotation sweep data file and computes offset

# to do: propagate and calculate new errors on field values
# create general script to take in a file and perform coord & field transformation

# sweep_out_2-8_good.txt:: four height values, four angle values, 5 trolley values
# plot horizontal slices of field

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

file = "/Users/alexandraklipfel/Desktop/senior_thesis/prelim_mapper_arm_data/sweep_out_2-8_good.txt"
dat = np.loadtxt(file)
print("Data Shape: ", dat.shape)
length = dat.shape[0]
print("Number of Data Points: ", length)

# number of vertical values
vertnum = 4

# set reference values
T0 = 200
R0 = 0 #don't know
V0 = 0 #don't know

# grab columns
#position
T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column
#field
BxProbe = dat[:, 3] #4th column
BxProbe = BxProbe - 0.15
BxProbeErr = dat[:, 4] #5th column
ByProbe = dat[:, 5] #6th column
ByProbeErr = dat[:, 6] #7th column
BzProbe = dat[:, 7] #4th column
BzProbe = BzProbe +0.66403
BzProbeErr = dat[:, 8] #5th column

rvals = (22.7 / 30000) * (T - T0)
#phivals = (np.pi / 8000) * (R - R0)
phivals = (np.pi / 8000) * (R - R0) - (np.pi/2)
zvals = (38.61 / 50000) * (V - V0)

# create transfromed array in lab frame
lab_dat = np.zeros(6)

for i in range(length):
    # transform to probe position to cartesian coords
    x = rvals[i] * np.cos(phivals[i])
    y = rvals[i] * np.sin(phivals[i])
    z = zvals[i]
    # transform B field with a rotation in probe xz plane
    #Bx = BzProbe[i] * np.cos(phivals[i]) + BxProbe[i] * np.sin(phivals[i])
    #By = BzProbe[i] * np.sin(phivals[i]) - BxProbe[i] * np.cos(phivals[i])
    Bx = BxProbe[i] * np.cos(phivals[i]) - BzProbe[i] * np.sin(phivals[i])
    By = BxProbe[i] * np.sin(phivals[i]) + BzProbe[i] * np.cos(phivals[i])
    Bz = -ByProbe[i]
    lab_dat = np.vstack([lab_dat, [x, y, z, Bx, By, Bz]])
    
# drop the first fow of zeros
lab_dat = lab_dat[1:, :]



# sort array by the z value (third column)
#lab_dat = lab_dat[np.argsort(lab_dat[:, 2])]

pd.DataFrame(lab_dat).to_csv("/Users/alexandraklipfel/Desktop/transformed_testdat.csv")

sub_length = int(length / vertnum)

'''for i in range(vertnum):
    # print out the vector field at that vertical value
    min = i * sub_length
    max = (i+1) * sub_length -1
    x = lab_dat[min:max, 0]
    y = lab_dat[min:max, 1]
    z = lab_dat[min:max, 2]
    Bx = lab_dat[min:max, 3]
    By = lab_dat[min:max, 4]
    Bz = lab_dat[min:max, 5]
    #plt.quiver(x, y, Bx, By)
    #plt.savefig("horizontal_cut" + str(i) + ".png")
    #plt.show()
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')

    #ax.quiver(x, y, z[0], Bx, By, Bz)
    #plt.savefig("3D_vf_" + str(i) + "_.png")
    #plt.show()'''
    

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.quiver(lab_dat[:,0], lab_dat[:,1], lab_dat[:,2], lab_dat[:,3], lab_dat[:,4], lab_dat[:,5])
#plt.savefig("3D_vf.png")
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.quiver(lab_dat[0:19,0], lab_dat[0:19,1], lab_dat[0,2], lab_dat[0:19,3], lab_dat[0:19,4], lab_dat[0:19,5], color="blue")
#ax.quiver(lab_dat[20:39,0], lab_dat[20:39,1], lab_dat[20,2], lab_dat[20:39,3], lab_dat[20:39,4], lab_dat[20:39,5], color="green")
#ax.quiver(lab_dat[40:59,0], lab_dat[40:59,1], lab_dat[40,2], lab_dat[40:59,3], lab_dat[40:59,4], lab_dat[40:59,5], color="orange")
#ax.quiver(lab_dat[60:79,0], lab_dat[60:79,1], lab_dat[60,2], lab_dat[60:79,3], lab_dat[60:79,4], lab_dat[60:79,5], color="red")
#plt.savefig("3D_vf.png")
#plt.show()

def remove_dat(data, axis, val1, val2, spread):
    #data: array of transformed data (2dim)
    #axis: 1, 2, or 3--axis that's varying (0==x, etc.)
    keep = np.array([])
    if axis == 0: #x is varying
        for i in range(len(data[:, axis])):
            if ((data[i, 1] > (val1 - spread)) and (lab_dat[i, 1] < (val1 + spread)) and (lab_dat[i, 2] > (val2 - spread)) and (lab_dat[i, 2] < (val2 + spread))):
                keep = np.append(keep, i)
                
    if axis == 1: #y is varying
        for i in range(len(data[:, axis])):
            if ((data[i, 0] > (val1 - spread)) and (lab_dat[i, 0] < (val1 + spread)) and (lab_dat[i, 2] > (val2 - spread)) and (lab_dat[i, 2] < (val2 + spread))):
                keep = np.append(keep, i)
                
    if axis == 2: #z is varying
        for i in range(len(data[:, axis])):
            if ((data[i, 0] > (val1 - spread)) and (lab_dat[i, 0] < (val1 + spread)) and (lab_dat[i, 1] > (val2 - spread)) and (lab_dat[i, 1] < (val2 + spread))):
                keep = np.append(keep, i)
    return keep

#print(lab_dat[:, 2])

indices1 = remove_dat(lab_dat, 0, 0, 524.3284, .01)
indices1 = indices1.astype(int)
indices2 = remove_dat(lab_dat, 0, 0, 541.8311, .01)
indices2 = indices2.astype(int)
indices3 = remove_dat(lab_dat, 0, 0, 559.3345, .01)
indices3 = indices3.astype(int)
indices4 = remove_dat(lab_dat, 0, 0, 576.8380, .01)
indices4 = indices4.astype(int)

print(lab_dat[indices1, 3])
print(BzProbe[indices1])

plt.scatter(lab_dat[indices1, 0], lab_dat[indices1, 3], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indices2, 0], lab_dat[indices2, 3], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indices3, 0], lab_dat[indices3, 3], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indices4, 0], lab_dat[indices4, 3], color="red", label="z=52.5 cm")
plt.xlabel("x coordinate")
plt.ylabel("Bx")
plt.title("Bx vs. x")
plt.legend(loc="upper left")
plt.savefig("Bx_vs_x.png")
plt.show()

plt.scatter(lab_dat[indices1, 0], lab_dat[indices1, 4], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indices2, 0], lab_dat[indices2, 4], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indices3, 0], lab_dat[indices3, 4], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indices4, 0], lab_dat[indices4, 4], color="red", label="z=52.5 cm")
plt.xlabel("x coordinate")
plt.ylabel("By")
plt.title("By vs. x")
plt.legend(loc="upper right")
plt.savefig("By_vs_x.png")
plt.show()

plt.scatter(lab_dat[indices1, 0], lab_dat[indices1, 5], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indices2, 0], lab_dat[indices2, 5], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indices3, 0], lab_dat[indices3, 5], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indices4, 0], lab_dat[indices4, 5], color="red", label="z=52.5 cm")
plt.xlabel("x coordinate")
plt.ylabel("Bz")
plt.title("Bz vs. x")
plt.legend(loc=[.6, .7])
plt.savefig("Bz_vs_x.png")
plt.show()

indicesy1 = remove_dat(lab_dat, 1, 0, 524.3284, .01)
indicesy1 = indicesy1.astype(int)
indicesy2 = remove_dat(lab_dat, 1, 0, 541.8311, .01)
indicesy2 = indicesy2.astype(int)
indicesy3 = remove_dat(lab_dat, 1, 0, 559.3345, .01)
indicesy3 = indicesy3.astype(int)
indicesy4 = remove_dat(lab_dat, 1, 0, 576.8380, .01)
indicesy4 = indicesy4.astype(int)

plt.scatter(lab_dat[indicesy1, 1], lab_dat[indicesy1, 3], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indicesy2, 1], lab_dat[indicesy2, 3], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indicesy3, 1], lab_dat[indicesy3, 3], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indicesy4, 1], lab_dat[indicesy4, 3], color="red", label="z=52.5 cm")
plt.xlabel("y coordinate")
plt.ylabel("Bx")
plt.title("Bx vs. y")
plt.legend(loc="lower left")
plt.savefig("Bx_vs_y.png")
plt.show()

plt.scatter(lab_dat[indicesy1, 1], lab_dat[indicesy1, 4], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indicesy2, 1], lab_dat[indicesy2, 4], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indicesy3, 1], lab_dat[indicesy3, 4], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indicesy4, 1], lab_dat[indicesy4, 4], color="red", label="z=52.5 cm")
plt.xlabel("y coordinate")
plt.ylabel("By")
plt.title("By vs. y")
plt.legend(loc="upper right")
plt.savefig("By_vs_y.png")
plt.show()

plt.scatter(lab_dat[indicesy1, 1], lab_dat[indicesy1, 5], color="blue", label="z=0 cm")
plt.scatter(lab_dat[indicesy2, 1], lab_dat[indicesy2, 5], color="green", label="z=17.5 cm")
plt.scatter(lab_dat[indicesy3, 1], lab_dat[indicesy3, 5], color="orange", label="z=35.0 cm")
plt.scatter(lab_dat[indicesy4, 1], lab_dat[indicesy4, 5], color="red", label="z=52.5 cm")
plt.xlabel("y coordinate")
plt.ylabel("Bz")
plt.title("Bz vs. y")
plt.legend(loc="upper right")
plt.savefig("Bz_vs_y.png")
plt.show()

'''indicesz1 = remove_dat(lab_dat, 2, 0, 0, .01)
indicesz1 = indicesz1.astype(int)

plt.scatter(lab_dat[indicesz1, 2], lab_dat[indicesz1, 3], color="blue", label="x=0, y=0")
plt.xlabel("z coordinate")
plt.ylabel("Bx")
plt.title("Bx vs. z")
plt.legend(loc="upper right")
plt.savefig("Bx_vs_z.png")
plt.show()

plt.scatter(lab_dat[indicesz1, 2], lab_dat[indicesz1, 4], color="blue", label="x=0, y=0")
plt.xlabel("z coordinate")
plt.ylabel("By")
plt.title("By vs. z")
plt.legend(loc="upper right")
plt.savefig("By_vs_z.png")
plt.show()

plt.scatter(lab_dat[indicesz1, 2], lab_dat[indicesz1, 5], color="blue", label="x=0, y=0")
plt.xlabel("z coordinate")
plt.ylabel("Bz")
plt.title("Bz vs. z")
plt.legend(loc="upper right")
plt.savefig("Bz_vs_z.png")
plt.show()'''
