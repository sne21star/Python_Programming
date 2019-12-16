import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import statistics
from statistics import mean
import randomcolor

def trainingMiddle(inputs, yB, strV):
    # Training and testing split, with 25% of the data reserved as the test set
    [B_train, B_test, b_train, b_test] = train_test_split(inputs, yB, test_size=.10, random_state=101)

    # Define the range of lambda to test
    lmbda = np.concatenate((np.arange(0.1,1.1,0.1), np.arange(1.5, 11, 0.5), np.arange(11, 101, 1)))

    MODEL = []
    MSE = []

    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(B_train, b_train, l)
        # Evaluate the MSE on the test set
        mse = error(B_test, b_test, model, strV)
        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    # Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]
    MSE_best = MSE_best[0]
    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))

    # Plot the MSE as a function of lmbda
    rand_color = randomcolor.RandomColor()
    if(strV != 'Total'):
        plt.plot(lmbda, MSE, color=str((rand_color.generate()[0])), label=strV)
    plt.title('Lambda v MSE for all Bridges')
    plt.xlabel('lmbda')
    plt.ylabel('mse')

    return model_best


def main():
    #Importing dataset
    bicycleData = pd.read_csv('bikeData.csv', thousands=',')
    # Feature and target matrices
    intPercipitation = []
    for x in bicycleData['Precipitation']:
        y = x.split()
        if(len(y) > 1):
            y = [y[0]]
        if(y[0] is 'T'):
            y = ['0.00']
        intPercipitation.append(float(y[0]))
    bicycleData['Precipitation'] = pd.Series(intPercipitation)

    inputs = bicycleData[['High Temp (°F)', 'Low Temp (°F)', 'Precipitation']]

    bicycleData.replace(',', '')

    #bicycleData[['Brooklyn Bridge']] = bicycleData[['Brooklyn Bridge']].astype(float)
    #bicycleData[['Manhattan Bridge']] = bicycleData[['Manhattan Bridge']].astype(float)
    #bicycleData[['Williamsburg Bridge']] = bicycleData[['Williamsburg Bridge']].astype(float)
    #bicycleData[['Queensboro Bridge']] = bicycleData[['Queensboro Bridge']].astype(float)
    #bicycleData[['Total']] = bicycleData[['Total']].astype(float)

    yB = bicycleData[['Brooklyn Bridge']]
    yM = bicycleData[['Manhattan Bridge']]
    yW = bicycleData[['Williamsburg Bridge']]
    yQ = bicycleData[['Queensboro Bridge']]
    #plt.legend(['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'])
    yT = bicycleData[['Total']]

    #Normalizing Data
    meanI = []
    sdI = []
    [inputs, meanI, sdI] = normalize(inputs)


    model_best = []

    model_best.append(trainingMiddle(inputs, yB, 'Brooklyn Bridge'))
    print('R^2 B')
    print(model_best[0].score(inputs, yB, sample_weight=None))


    model_best.append(trainingMiddle(inputs, yW, 'Williamsburg Bridge'))
    print('R^2 W')
    print(model_best[1].score(inputs, yW, sample_weight=None))

    model_best.append(trainingMiddle(inputs, yQ, 'Queensboro Bridge'))
    print('R^2 Q')
    print(model_best[2].score(inputs, yQ, sample_weight=None))

    model_best.append(trainingMiddle(inputs, yM, 'Manhattan Bridge'))
    print('R^2 M')
    print(model_best[3].score(inputs, yM, sample_weight=None))

    model_best.append(trainingMiddle(inputs, yT, 'Total'))

    plt.legend()


    plt.show()

    return model_best, meanI, sdI


#Function that normalizes features to zero mean and unit variance.
#Input: Feature matrix X.
#Output: X, the normalized version of the feature matrix.
def normalize(X):
    X = X.copy()
    meanX = []
    sdX = []
    for c in X.columns:
        X.loc[:,c] = (X.loc[:,c] - X[c].mean()) / (X[c].std(ddof=0.00001))
        meanX.append(X[c].mean())
        sdX.append((X[c].std(ddof=0.00001)))
    return X, meanX, sdX

#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    model = linear_model.Ridge(alpha=l, fit_intercept=True)
    model.fit(X, y)
    return model

#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model, strV):
    yHat = model.predict(X)
    yList = y[strV].values
    yLen = len(yList)
    rangeValues = range(0, yLen)
    mseSub = [(pow((yList[indexX] - yHat[indexX]),2)) for indexX in rangeValues]
    mseva = sum(mseSub) / yLen
    return mseva
def predictionCase(model_Fit, testCase):

    testCaseMean = mean(testCase)
    testCaseSd = statistics.stdev(testCase)

    testCase = [[(i - testCaseMean) / testCaseSd for i in testCase]]
    normalizeValue = [(i - j) / k for i, j, k in zip(testCase, meanX, sdX)]
    valueB = model_Fit[0].predict(normalizeValue)
    valueW = model_Fit[1].predict(normalizeValue)
    valueQ = model_Fit[2].predict(normalizeValue)
    valueM = model_Fit[3].predict(normalizeValue)
    valueT = model_Fit[4].predict(normalizeValue)

    print("Predict riders on Brooklyn " + str(round((valueB[0][0]), 0)))
    #print("Unrounded Version: " + str(valueB[0][0]))

    print("Predict riders on  Williamsburg " + str(round((valueW[0][0]), 0)))
    #print("Unrounded Version: " + str(valueW[0][0]))

    print("Predict riders on  Queensboro " + str(round((valueQ[0][0]), 0)))
    #print("Unrounded Version: " + str(valueQ[0][0]))

    print("Predict riders on  Manhattan " + str(round((valueM[0][0]), 0)))
    #print("Unrounded Version: " + str(valueM[0][0]))

    print("Predict riders Total " + str(round((valueT[0][0]), 0)))
    # print("Unrounded Version: " + str(valueM[0][0]))

    print("")
    return [round((valueB[0][0]), 0), round((valueW[0][0]), 0), round((valueQ[0][0]), 0), round((valueM[0][0]), 0), round((valueT[0][0]), 0)]

if __name__ == '__main__':
    [model_Fit, meanX, sdX] = main()
    coefficientsB = model_Fit[0].coef_
    coefficientsW = model_Fit[1].coef_
    coefficientsQ = model_Fit[2].coef_
    coefficientsM = model_Fit[3].coef_
    coefficientsT = model_Fit[4].coef_


    print()
    print('Coefficients')
    print('Brooklyn Bridge: ' + str(coefficientsB))
    print('Williamsburg Bridge: ' + str(coefficientsW))
    print('Queensboro Bridge: ' + str(coefficientsQ))
    print('Manhattan Bridge: ' + str(coefficientsM))
    print('Total: ' + str(coefficientsT))
    print()
    #What are some test cases
    #Highest recorded temperature -> 106
    #Lowest recorded temperature -> -15
    #2 springs
    #2 summers
    #2 fall
    #average high temp = Annual high temperature:	62.3°F
    # average low temp = Annual low temperature:	48°F
    #average different = 14.3
    #https://www.currentresults.com/Weather/New-York/Places/new-york-city-temperatures-by-month-average.php

    averageDiff = 14.3
    testCase = [-15 + averageDiff, -15, .4] #lowest temperature
    testCase1 = [106, 106 - averageDiff, .4] #highest temperature
    testCase2 = [50, 35, .4] #Spring
    testCase3 = [61, 45, .4] #Spring
    testCase4 = [84, 69, .4] #Summer
    testCase5 = [83, 68, .4] #Summer
    testCase6 = [40, 40 - averageDiff, .4] #Fall
    testCase7 = [35, 35 - averageDiff, .4] #Fall
    testCase8 = [35, 35, .4] #random, no difference between temperature
    testCase9 = [45, 45, .4]
    testCase10 = [65, 65, .4]
    testCase11 = [90, 90, .4]

    numberRiderB = []
    numberRiderQ = []
    numberRiderW = []
    numberRiderM = []
    numberRiderT = []

    print('Test Case format: ' + '[Highest Temperature, Lowest Temperature, Percipitation range[0,1]')
    print()

    print('testCase ' + str(testCase))
    x = predictionCase(model_Fit, testCase)

    print('testCase1 ' + str(testCase1))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])

    x = predictionCase(model_Fit, testCase1)

    print('testCase2 ' + str(testCase2))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase2)
    print('testCase3 ' + str(testCase3))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase3)
    print('testCase4 ' + str(testCase4))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase4)
    print('testCase5 ' + str(testCase5))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase5)
    print('testCase6 ' + str(testCase6))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase6)
    print('testCase7 ' + str(testCase7))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase7)
    print('testCase8 ' + str(testCase8))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase8)
    print('testCase9 ' + str(testCase9))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase9)
    print('testCase10 ' + str(testCase10))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    x = predictionCase(model_Fit, testCase10)


    print('testCase11 ' + str(testCase11))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    predictionCase(model_Fit, testCase11)

    testCase12 = [78.1, 66, .01]
    testCase13 = [43.0, 37.9, .09]
    testCase14 = [63.0, 46.9, 0] #10/10,Monday,63.0,46.9,0.00,"2,838","5,205","6,054","3,763","17,860"
    testCase15 = [63.0,46.9,0.00] #"3,184","6,201","7,227","4,334","20,946"

    print('testCase12 ' + str(testCase12))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    predictionCase(model_Fit, testCase12)

    print('testCase13 ' + str(testCase13))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    predictionCase(model_Fit, testCase13)

    print('testCase14 ' + str(testCase14))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    predictionCase(model_Fit, testCase14)

    print('testCase15 ' + str(testCase15))
    numberRiderB.append(x[0])
    numberRiderW.append(x[1])
    numberRiderQ.append(x[2])
    numberRiderM.append(x[3])
    numberRiderT.append(x[4])
    predictionCase(model_Fit, testCase15)

    rand_color = randomcolor.RandomColor()

    testCaselen = range(0,15)

    #4/1,Friday,78.1,66,0.01,"1,704","3,126","4,115","2,552","11,497"
    #4/9,Saturday,43.0,37.9,0.09,504,997,"1,507","1,502","4,510"
    #6/5,Sunday,70.0,64.9,0.91,918,"1,593","2,473","1,805","6,789"
    #8/7,Sunday,88.0,71.1,0.00,"3,526","4,430","5,364","4,026","17,346"

    #10/17,Monday,80.1,63.0,0.00,"3,465","6,719","7,731","4,662","22,577"
    #10/18,Tuesday,81.0,66.9,0.00,"4,029","7,594","8,761","5,098","25,482"



    plt.show()