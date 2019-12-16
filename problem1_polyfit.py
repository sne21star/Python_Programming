import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of n = degrees[i].
def main(x, y, degrees):
    paramFits = []
    for n in degrees:
        X = feature_matrix(x, n)
        B = least_squares(X, y)
        paramFits.append(B)
    return paramFits


#Return the feature matrix for fitting a polynomial of degree n based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and n as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by n+1.
def feature_matrix(x, n):
    X = np.zeros([len(x), n+1])
    columnDegree = []
    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^n, x^(n-1), ..., x^0.
    while n > -1:
        columnDegree.append([pow(x_i, n) for x_i in x])
        n = n - 1
    X = np.column_stack(columnDegree)
    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    # (
    # B = np.linalg.inv((np.transpose(X)*X))*np.transpose(X)*y
    C = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    lenB = len(C)
    indexC = 0
    B = []
    while indexC < lenB:
        # C[indexC] = [round(x, 6) for x in C[indexC]]
        B.append(round(C[indexC],6))
        indexC = indexC + 1
    return B
def printParam(paramFits):
    print("linear: " + str(paramFits[0][0]) + "X + " + str(paramFits[0][1]))
    print("Quadratic: " + str(paramFits[1][0]) + "X^2 + " + str(paramFits[1][1]) + "X + " + str(paramFits[1][2]))
    print("Cubic: " + str(paramFits[2][0]) + "X^3 + " + str(paramFits[2][1]) + "X^2 + " + str(paramFits[2][2]) + "X + " + str(paramFits[2][3]))
    print("Quartic: " + str(paramFits[3][0]) + "X^4 + " + str(paramFits[3][1]) + "X^3 + " + str(paramFits[3][2]) + "X^2 + " + str(paramFits[3][3]) + "X + " + str(paramFits[3][4]))
    print("Quintic: " + str(paramFits[4][0]) + "X^5 + " + str(paramFits[4][1]) + "X^4 + " + str(paramFits[4][2]) + "X^3 + " + str(paramFits[4][3]) + "X^2 + " + str(paramFits[4][4]) + "X + " + str(paramFits[4][5]))
    print("")
    return
def plottingData(paramFits, xData, yData, bridgeName):

    xD = []
    for x in xData:
        xD.append(x[0])
    plt.scatter(xData, yData, label='Original', c='black', alpha=0.5)
    firstDegree = [k * paramFits[0][0] + paramFits[0][1] for k in xData]
    secondDegree = [k ** 2 * paramFits[1][0] + paramFits[1][1] * k + paramFits[1][2] for k in xData]
    thirdDegree = [k ** 3 * paramFits[2][0] + paramFits[2][1] * k ** 2 + paramFits[2][2] * k + paramFits[2][3] for k in
                   xData]
    fourthDegree = [k ** 4 * paramFits[3][0] + paramFits[3][1] * k ** 3 + paramFits[3][2] * k ** 2 + paramFits[3][3] * k + paramFits[3][4] for k in xData]
    fifthDegree = [k ** 5 * paramFits[4][0] + paramFits[4][1] * k ** 4 + paramFits[4][2] * k ** 3 + paramFits[4][3] * k ** 2 + paramFits[4][4] * k + paramFits[4][5] for k in xData]
    line, = plt.plot(xData, firstDegree, label='Linear', color='blue')
    x2, y2 = zip(*sorted(zip(xData, secondDegree), key=lambda x: x[0]))
    plt.plot((x2), y2, label='Quadratic', color='orange', linestyle='-')
    x2, y2 = zip(*sorted(zip(xData, thirdDegree), key=lambda x: x[0]))
    plt.plot((x2), (y2), label='Cubic', color='red')
    x2, y2 = zip(*sorted(zip(xData, fourthDegree), key=lambda x: x[0]))
    plt.plot(((x2)), (y2), label='Quartic', color='teal')
    x2, y2 = zip(*sorted(zip(xData, fifthDegree), key=lambda x: x[0]))
    plt.plot((x2), (y2), label='Quintic', color='violet')
    plt.legend()
    plt.title('NYC Bicycle Data on ' + bridgeName + ' Bridge')
    plt.xlabel('High Temperature F˚')
    plt.ylabel('Number of Cyclists on ')

if __name__ == '__main__':
    bicycleData = pd.read_csv('bikeDATA.csv')

    brooklyn = bicycleData[['Brooklyn Bridge']]
    brooklyn = brooklyn.values
    Br = []
    for x in brooklyn:
        Br.append(float(x[0].replace(',', '')))
    print(sum(Br))

    manhattan = bicycleData[['Manhattan Bridge']]
    manhattan = manhattan.values
    M = []
    for x in manhattan:
        M.append(float(x[0].replace(',', '')))
    print(sum(M))

    williamsburg = bicycleData[['Williamsburg Bridge']]
    williamsburg = williamsburg.values
    W = []
    for x in williamsburg:
        W.append(float(x[0].replace(',', '')))
    print(sum(W))

    queensboro = bicycleData[['Queensboro Bridge']]
    queensboro = queensboro.values
    Q = []
    for x in queensboro:
        Q.append(float(x[0].replace(',', '')))
    print(sum(Q))

    total = bicycleData[['Total']]
    total = total.values
    y = []

    lowTemp = bicycleData[['Low Temp (°F)']]
    lowTemp = lowTemp.values

    highTemp = bicycleData[['High Temp (°F)']]
    highTemp = highTemp.values

    for x in total:
        y.append(float(x[0].replace(',', '')))

    timeList = range(0,len(queensboro))

    degrees = [1, 2, 3, 4, 5]
    fig1 = plt.figure(1)

    paramFits = main(highTemp, M, degrees)
    printParam(paramFits)
    plottingData(paramFits, highTemp, M, 'Manhattan')

    fig2 = plt.figure(2)
    paramFits = main(highTemp, Br, degrees)
    printParam(paramFits)
    plottingData(paramFits, highTemp, Br,  'Brooklyn')

    fig3 = plt.figure(3)
    paramFits = main(highTemp, Q, degrees)
    printParam(paramFits)
    plottingData(paramFits, highTemp, Q, 'Queensboro')

    fig4 = plt.figure(4)

    paramFits = main(highTemp, W, degrees)
    printParam(paramFits)
    plottingData(paramFits, highTemp, W, 'Williamsburg')

    plt.show()
    fig1.savefig('Manhattan.png')
    fig2.savefig('Brooklyn.png')
    fig3.savefig('Queensboro.png')
    fig4.savefig('Williamsburg.png')
