import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Part 1
# Loading Data
def boston_factors():
    data = load_boston()
    print(data.feature_names)

    # Converting into a Dataframe for easier use
    boston = pd.DataFrame(data.data, columns=data.feature_names)

    # Median Value is your target variable
    target = data.target

    # Linear Regression Results
    # Split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(boston, target, test_size=0.2, random_state=4)
    lm = sm.OLS(y_train, x_train)  # OLS
    model = lm.fit()
    print(model.summary())

    # Based on the model, the factors with the largest coefficients will have the biggest impacts
    # on the median value of a house
    coefficients = pd.DataFrame(model.params, columns=["Coef"])
    coefficients['Coef']=coefficients['Coef'].abs()
    coefficients = coefficients.sort_values('Coef', ascending=False)
    print(coefficients)

# Factor with greatest impact on value: RM - Average number of rooms per dwelling
# Factor with least impact on vale: TAX - Full-value property-tax rate per $10,000

# Part 2


def elbows():
    iris = load_iris()
    print(iris.feature_names)
    iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Initialize Kmeans and fit the data
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(iris)
    print(kmeans.inertia_)

    # Since inertia is the sum of squared distances of samples to their closest cluster center, we can use this value
    # to plot the elbow to find the optimal number of clusters

    inertia_iris = []
    clusters = range(1, 10)
    for i in clusters:
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(iris)
        inertia_iris.append(kmeans.inertia_)

    plt.plot(clusters, inertia_iris)
    plt.title("Elbow Heuristic: Iris Dataset")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

    # Based on this graph, we can see that the inertia value is dropping by a lot up until 3 clusters, where after
    # it is only falling by a little bit. Based on this, we see that the optimal number of clusters is 3

    wine = load_wine()
    wine = pd.DataFrame(wine.data, columns=wine.feature_names)

    inertia_wine = []
    for i in clusters:
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(wine)
        inertia_wine.append(kmeans.inertia_)

    plt.plot(clusters, inertia_wine)
    plt.title("Elbow Heuristic: Wine Dataset")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

    # We see the same thing here, the inertia value falls rapidly until we hit 3 clusters, where it begins to level off


if __name__ == "__main__":
    boston_factors()
    elbows()
