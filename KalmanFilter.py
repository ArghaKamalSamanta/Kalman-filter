import numpy as np
import ast
import matplotlib.pyplot as plt


# Function for estimation
def prediction(xt_1, pt_1, ut):
    A = np.array([[1, 0, 0.01, 0],
                  [0, 1, 0, 0.01],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[0.0005, 0],
                  [0, 0.0005],
                  [0.01, 0],
                  [0, 0.01]])

    Xt = np.dot(A, xt_1) + np.dot(B, ut)

    Q = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 100, 100, 100],
                  [0, 100, 0, 100]])

    Pt = np.dot(A, np.dot(pt_1, A.T)) + Q

    return Xt, Pt


# Function for update step
def update(Xt, Pt, xmt):
    R = np.array([[300, 100, 100, 100],
                  [100, 400, 100, 100],
                  [100, 100, 500, 100],
                  [100, 100, 100, 700]])

    kt = np.dot(Pt, np.linalg.inv(Pt + R))

    xt = Xt + np.dot(kt, (xmt - Xt))

    I = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    pt = np.dot((I - kt), Pt)

    return xt, pt


# Opening the given file and reading values from that file
f = open('kalmann.txt')
content = f.readlines()

sigma_x, sigma_y, sigma_vx, sigma_vy = 0.00000000000000000007, \
                                       0.00000000000000000008, \
                                       0.00000008, \
                                       0.00000005

# Initial error matrix in the estimation step
p0 = np.array([[sigma_x ** 2, sigma_x * sigma_y, sigma_x * sigma_vx, sigma_x * sigma_vy],
               [sigma_y * sigma_x, sigma_y ** 2, sigma_y * sigma_vx, sigma_y * sigma_vy],
               [sigma_x * sigma_vx, sigma_vx * sigma_y, sigma_vx ** 2, sigma_vx * sigma_vy],
               [sigma_x * sigma_vy, sigma_vy * sigma_y, sigma_vy * sigma_vx, sigma_vy ** 2]])


updated_position = []
measured_position = []

updated_velocity = []
measured_velocity = []

for i in range(len(content) - 1):
    arr1 = ast.literal_eval(content[i])
    arr2 = ast.literal_eval(content[i + 1])

    if i == 0:
        arr1 = list(arr1)
        arr1.append(0.01)
        arr1.append(6.33)
        arr1 = tuple(arr1)

    xmt = np.resize(arr1, (4, 1))
    x0 = np.array([[int(arr1[0])],
                   [int(arr1[1])],
                   [int(arr1[2])],
                   [int(arr1[3])]])

    u = np.array([[arr2[2] - arr1[2]],
                  [arr2[3] - arr1[3]]])

    p0_temp = p0

    for j in range(100):
        x0, p0_temp = prediction(x0, p0_temp, u)
        x0, p0_temp = update(x0, p0_temp, xmt)

    print("At t =", i + 1, "sec, position = (", x0[0][0], ", ", x0[1][0], ")")
    print("Vx =", x0[2][0], "Vy =", x0[3][0])
    print("The uncertainty matrix:")
    print(p0_temp)
    print("----------------------------------------------------------------")

    updated_position.append((x0[0][0], x0[1][0]))
    measured_position.append((arr1[0], arr1[1]))

    updated_velocity.append((x0[2][0], x0[3][0]))
    measured_velocity.append((arr1[2], arr1[3]))

# Plotting various graphs
for i in range(len(updated_position) - 1):

    x_values = [updated_position[i][0], updated_position[i + 1][0]]
    y_values = [updated_position[i][1], updated_position[i + 1][1]]
    vx_values = [updated_velocity[i][0], updated_velocity[i + 1][0]]
    vy_values = [updated_velocity[i][1], updated_velocity[i + 1][1]]
    time = [i, i + 1]

    # Plot of position_y vs position_x
    plt.figure("y v/s x")
    plt.plot(x_values, y_values, 'green')
    plt.plot(measured_position[i][0], measured_position[i][1], marker="o", markersize=0.5, markeredgecolor="red",
             markerfacecolor="red")
    plt.text(173, -310, "Green: Kalman filtered data\nRed: Measured data")

    # Plot of position_x vs time
    plt.figure("x v/s time")
    plt.plot(time, x_values, 'green')
    plt.plot(i, measured_position[i][0], marker="o", markersize=0.5, markeredgecolor="red",
             markerfacecolor="red")
    plt.text(232, -310, "Green: Kalman filtered data\nRed: Measured data")

    # Plot of position_y vs time
    plt.figure("y v/s time")
    plt.plot(time, y_values, 'green')
    plt.plot(i, measured_position[i][1], marker="o", markersize=0.5, markeredgecolor="red",
             markerfacecolor="red")
    plt.text(232, 100, "Green: Kalman filtered data\nRed: Measured data")

    # Plot of velocity_x vs time
    plt.figure("vx v/s time")
    plt.plot(time, vx_values, 'green')
    plt.plot(i, measured_velocity[i][0], marker="o", markersize=0.5, markeredgecolor="red",
             markerfacecolor="red")
    plt.text(200, -6, "Green: Kalman filtered data\nRed: Measured data")

    # Plot of velocity_y vs time
    plt.figure("vy v/s time")
    plt.plot(time, vy_values, 'green')
    plt.plot(i, measured_velocity[i][1], marker="o", markersize=0.5, markeredgecolor="red",
             markerfacecolor="red")
    plt.text(225, -6, "Green: Kalman filtered data\nRed: Measured data")

# Showing all the plots
plt.show()

# Closing the file
f.close()
