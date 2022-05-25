# imports rotation sweep data file and computes offset

# to do: propagate and calculate new errors on field values
# create general script to take in a file and perform coord & field transformation

# sweep_out_2-8_good.txt:: four height values, four angle values, 5 trolley values
# plot horizontal slices of field

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

file = "/Users/alexandraklipfel/Desktop/senior_thesis/prelim_mapper_arm_data/sweep_2-25_out_no300.txt"
dat = np.loadtxt(file)
print("Data Shape: ", dat.shape)
length = dat.shape[0]
print("Number of Data Points: ", length)

# set reference values
T0 = 200
R0 = 2500
V0 = min(dat[:, 2])

# grab columns
#position
T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column
#field
BxProbe = dat[:, 3] #4th column
BxProbe = BxProbe
BxProbeErr = dat[:, 4] #5th column
ByProbe = dat[:, 5] #6th column
ByProbeErr = dat[:, 6] #7th column
BzProbe = dat[:, 7] #4th column
BzProbe = BzProbe
BzProbeErr = dat[:, 8] #5th column

rvals = (22.7 / 30000) * (T - T0)
#phivals = (np.pi / 8000) * (R - R0)
phivals = (np.pi / 8000) * (R - R0) + (np.pi/2)
zvals = (38.61 / 50000) * (V - V0)
thetavals = (np.pi / 8000) * (R - R0)

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
    Bx = BxProbe[i] * np.cos(thetavals[i]) - BzProbe[i] * np.sin(thetavals[i])
    By = BxProbe[i] * np.sin(thetavals[i]) + BzProbe[i] * np.cos(thetavals[i])
    Bz = -ByProbe[i]
    lab_dat = np.vstack([lab_dat, [x, y, z, Bx, By, Bz]])
    
# drop the first fow of zeros
lab_dat = lab_dat[1:, :]



# sort array by the z value (third column)
#lab_dat = lab_dat[np.argsort(lab_dat[:, 2])]

pd.DataFrame(lab_dat).to_csv("/Users/alexandraklipfel/Desktop/transformed_testdat.csv")


plt.quiver(lab_dat[0:55, 0], lab_dat[0:55, 1], lab_dat[0:55, 3], lab_dat[0:55, 4])
plt.savefig("horizontal_cut_" + str(0) + ".png")
plt.show()

plt.quiver(lab_dat[0:55, 0], lab_dat[0:55, 1], BxProbe[0:55], ByProbe[0:55])
plt.savefig("horizontal_cut_" + str(0) + ".png")
plt.show()

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

heights = np.unique(V)
print(heights)
num = len(heights)

def remove_dat(data, Rval, Vval, spread):
    #data: array of transformed data (2dim)
    keep = np.array([])
    for i in range(len(R)):
        if (((R[i] == Rval) or (R[i] == Rval + 8000)) and (V[i] > (Vval - spread)) and (V[i] < (Vval + spread))):
            keep = np.append(keep, i)
    return keep
    
def get_rad_dist(index_list):
    dist_list = np.zeros(len(index_list))
    for i in range(len(index_list)):
        x = lab_dat[index_list[i], 0]
        y = lab_dat[index_list[i], 1]
        dist = np.sqrt(x**2 + y**2)
        if i < (len(index_list) / 2):
            dist = -dist
        dist_list[i] = dist
    return dist_list
    
def get_vals(data, Baxis, angle, height, range):
    indices = remove_dat(data, angle, height, range)
    indices = indices.astype(int)
    x = get_rad_dist(indices)
    print(x)
    if Baxis == "x":
        y = lab_dat[indices, 3]
    if Baxis == "y":
        y = lab_dat[indices, 4]
    if Baxis == "z":
        y = lab_dat[indices, 5]
    print(y)
    return x, y
    
def get_mags(data, angle, height, spread):
    indices = remove_dat(data, angle, height, spread)
    indices = indices.astype(int)
    x = get_rad_dist(indices)
    y = np.zeros(len(indices))
    for i in range(len(indices)):
        y[i] = np.sqrt(data[indices[i], 3]**2 + data[indices[i], 4]**2 + data[indices[i], 5])
    return x, y
    
    
x1 = get_mags(lab_dat, 0., heights[0], 0.01)[0]
y1 = get_mags(lab_dat, 0., heights[0], 0.01)[1]

x2 = get_mags(lab_dat, 0., heights[1], 0.01)[0]
y2 = get_mags(lab_dat, 0., heights[1], 0.01)[1]

x3 = get_mags(lab_dat, 0., heights[2], 0.01)[0]
y3 = get_mags(lab_dat, 0., heights[2], 0.01)[1]

x4 = get_mags(lab_dat, 0., heights[3], 0.01)[0]
y4 = get_mags(lab_dat, 0., heights[3], 0.01)[1]

x5 = get_mags(lab_dat, 0., heights[4], 0.01)[0]
y5 = get_mags(lab_dat, 0., heights[4], 0.01)[1]

x6 = get_mags(lab_dat, 0., heights[5], 0.01)[0]
y6 = get_mags(lab_dat, 0., heights[5], 0.01)[1]

x7 = get_mags(lab_dat, 0., heights[6], 0.01)[0]
y7 = get_mags(lab_dat, 0., heights[6], 0.01)[1]

x8 = get_mags(lab_dat, 0., heights[7], 0.01)[0]
y8 = get_mags(lab_dat, 0., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("|B|")
plt.title("|B| vs distance at phi=56 deg.")
plt.savefig("B_0.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()
    
################################
    
x1 = get_vals(lab_dat, "y", 0., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "y", 0., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "y", 0., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "y", 0., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "y", 0., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "y", 0., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "y", 0., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "y", 0., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "y", 0., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "y", 0., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "y", 0., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "y", 0., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "y", 0., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "y", 0., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "y", 0., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "y", 0., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("By")
plt.title("By vs distance at phi=56 deg.")
plt.savefig("By_0.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

################################

x1 = get_vals(lab_dat, "x", 0., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "x", 0., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "x", 0., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "x", 0., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "x", 0., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "x", 0., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "x", 0., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "x", 0., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "x", 0., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "x", 0., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "x", 0., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "x", 0., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "x", 0., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "x", 0., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "x", 0., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "x", 0., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("Bx")
plt.title("Bx vs distance at phi=56 deg.")
plt.savefig("Bx_0.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

################################

x1 = get_vals(lab_dat, "z", 0., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "z", 0., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "z", 0., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "z", 0., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "z", 0., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "z", 0., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "z", 0., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "z", 0., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "z", 0., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "z", 0., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "z", 0., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "z", 0., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "z", 0., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "z", 0., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "z", 0., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "z", 0., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("Bz")
plt.title("Bz vs distance at phi=56 deg.")
plt.savefig("Bz_0.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

#################################
#################################

x1 = get_vals(lab_dat, "y", 4000., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "y", 4000., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "y", 4000., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "y", 4000., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "y", 4000., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "y", 4000., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "y", 4000., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "y", 4000., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "y", 4000., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "y", 4000., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "y", 4000., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "y", 4000., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "y", 4000., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "y", 4000., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "y", 4000., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "y", 4000., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("By")
plt.title("By vs distance at phi=146 deg.")
plt.savefig("By_4000.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

################################

x1 = get_vals(lab_dat, "x", 4000., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "x", 4000., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "x", 4000., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "x", 4000., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "x", 4000., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "x", 4000., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "x", 4000., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "x", 4000., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "x", 4000., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "x", 4000., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "x", 4000., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "x", 4000., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "x", 4000., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "x", 4000., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "x", 4000., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "x", 4000., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("Bx")
plt.title("Bx vs distance at phi=146 deg.")
plt.savefig("Bx_4000.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

################################

x1 = get_vals(lab_dat, "z", 4000., heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "z", 4000., heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "z", 4000., heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "z", 4000., heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "z", 4000., heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "z", 4000., heights[2], 0.01)[1]

x4 = get_vals(lab_dat, "z", 4000., heights[3], 0.01)[0]
y4 = get_vals(lab_dat, "z", 4000., heights[3], 0.01)[1]

x5 = get_vals(lab_dat, "z", 4000., heights[4], 0.01)[0]
y5 = get_vals(lab_dat, "z", 4000., heights[4], 0.01)[1]

x6 = get_vals(lab_dat, "z", 4000., heights[5], 0.01)[0]
y6 = get_vals(lab_dat, "z", 4000., heights[5], 0.01)[1]

x7 = get_vals(lab_dat, "z", 4000., heights[6], 0.01)[0]
y7 = get_vals(lab_dat, "z", 4000., heights[6], 0.01)[1]

x8 = get_vals(lab_dat, "z", 4000., heights[7], 0.01)[0]
y8 = get_vals(lab_dat, "z", 4000., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("Bz")
plt.title("Bz vs distance at phi=146 deg.")
plt.savefig("Bz_4000.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()

########################

x1 = get_mags(lab_dat, 4000., heights[0], 0.01)[0]
y1 = get_mags(lab_dat, 4000., heights[0], 0.01)[1]
print(x1)
print(y1)

x2 = get_mags(lab_dat, 4000., heights[1], 0.01)[0]
y2 = get_mags(lab_dat, 4000., heights[1], 0.01)[1]

x3 = get_mags(lab_dat, 4000., heights[2], 0.01)[0]
y3 = get_mags(lab_dat, 4000., heights[2], 0.01)[1]

x4 = get_mags(lab_dat, 4000., heights[3], 0.01)[0]
y4 = get_mags(lab_dat, 4000., heights[3], 0.01)[1]

x5 = get_mags(lab_dat, 4000., heights[4], 0.01)[0]
y5 = get_mags(lab_dat, 4000., heights[4], 0.01)[1]

x6 = get_mags(lab_dat, 4000., heights[5], 0.01)[0]
y6 = get_mags(lab_dat, 4000., heights[5], 0.01)[1]

x7 = get_mags(lab_dat, 4000., heights[6], 0.01)[0]
y7 = get_mags(lab_dat, 4000., heights[6], 0.01)[1]

x8 = get_mags(lab_dat, 4000., heights[7], 0.01)[0]
y8 = get_mags(lab_dat, 4000., heights[7], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))
plt.scatter(x4, y4, color="green", label="z=" + str((38.61 / 50000) * (heights[3] - V0)))
plt.scatter(x5, y5, color="blue", label="z=" + str((38.61 / 50000) * (heights[4] - V0)))
plt.scatter(x6, y6, color="purple", label="z=" + str((38.61 / 50000) * (heights[5] - V0)))
plt.scatter(x7, y7, color="black", label="z=" + str((38.61 / 50000) * (heights[6] - V0)))
plt.scatter(x8, y8, color="gray", label="z=" + str((38.61 / 50000) * (heights[7] - V0)))
plt.xlabel("radial distance")
plt.ylabel("|B|")
plt.title("|B| vs distance at phi=146 deg.")
plt.savefig("B_40000.png")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.show()
