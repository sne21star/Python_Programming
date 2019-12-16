import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import randomcolor

def main():

    # Importing dataset
    bicycleData = pd.read_csv('bikeData.csv', thousands=',')

    # Feature and target matrices
    intPrecipitation = []
    for x in bicycleData['Precipitation']:
        y = x.split()
        if(len(y) > 1):
            y = [y[0]]
        if(y[0] is 'T'):
            y = ['0.00']
        intPrecipitation.append(float(y[0]))
    bicycleData['Precipitation'] = pd.Series(intPrecipitation)

    bicycleData[['Total']] = bicycleData[['Total']].astype(str)
    bicycleData.replace(',', '')
    bicycleData[['Total']] = bicycleData[['Total']].astype(float)

    bicycleData.Total = bicycleData.Total.astype(float)

    X = bicycleData['Total'] - (bicycleData['Brooklyn Bridge'])
    print(X)

    bicycleData['Precipitation'] = bicycleData['Precipitation'].mask(bicycleData['Precipitation'] > 0.1, 1)
    bicycleData['Precipitation'] = bicycleData['Precipitation'].mask(bicycleData['Precipitation'] <= 0.1 , 0)

    y = bicycleData[['Precipitation']]
    print(y)
    print(sum(bicycleData['Williamsburg Bridge']))

    # Normalizing Data
    X , averages, standard_devs = normalize(X.to_frame())

    # Training and testing split, with 25% of the data reserved as the test set
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.10, random_state=101)

    # Define the range of lambda to test
    lmbda = []
    for i in np.arange(0.1, 1.1, 0.1):
        lmbda.append(i)
    for i in np.arange(1.5, 10.5, 0.5):
        lmbda.append(i)
    for i in np.arange(11, 101, 1):
        lmbda.append(i)
    # lmbda = [1, 100]

    # print(lmbda)
    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    # Plot the MSE as a function of lmbda
    plt.plot(lmbda, MSE)  # fill in
    plt.xlabel("lambda value")
    plt.ylabel("MSE")
    # plt.show()

    # Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]
    print(model_best.coef_)
    new_input = [1000]
    for i in range(0, len(new_input)):
        new_input[i] = (new_input[i] - averages[i]) / standard_devs[i]
    new_input = np.array(new_input).reshape(1, -1)


    r2_value = calc_r2(X_test, y_test, model)
    print("R^2 value:", r2_value)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    prediction = model_best.predict(new_input)
    print('prediction', prediction)
    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))

    # Plot the MSE as a function of lmbda
    rand_color = randomcolor.RandomColor()

    plt.plot(lmbda, MSE, color=str((rand_color.generate()[0])))
    plt.title('Lambda v MSE')
    plt.xlabel('lmbda')
    plt.ylabel('mse')
    # plt.show()

    plt.plot(np.sort(y_test))
    # plt.show()

    return model_best


#Function that normalizes features to zero mean and unit variance.
#Input: Feature matrix X.
#Output: X, the normalized version of the feature matrix.
def normalize(X):
    X = X.copy()
    averages= []
    standard_devs = []
    for col in X.columns:
         averages.append(X[col].mean())
         standard_devs.append(X[col].std())
         X[col] = (X[col] - X[col].mean()) / (np.std(X[col], ddof=0))
    # X = (X - X.stack().mean()) / X.stack().std()
    return X, averages, standard_devs


#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    #fill in
    model_ridge = linear_model.LogisticRegression(fit_intercept=True)
    model_ridge.fit(X,y)
    return model_ridge


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

def calc_r2(X,y,model):

    y_pred = model.predict(X)
    r2_value = r2_score(y, y_pred)
    return r2_value


if __name__ == '__main__':
    model_Fit = main()
    coefficients = model_Fit.coef_
    print('coef', coefficients)
