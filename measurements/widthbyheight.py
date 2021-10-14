import pandas
import numpy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope

# feed in and plot the data to see if there are any patterns
'''
From the visualization of the plot there appears to be four distinct areas of data with two possible outliers
I chose not to scale the data in this case since the points seemed to fall within relatively small range.
This could be an option however in fine tuning this model
'''
measures_df = pandas.read_excel('wxh.xlsx')
measures_df.plot(x='Width', y='Height', kind='scatter', xlim=(20, 32), ylim=(20, 32))
plt.savefig('raw_data.png')
plt.show()


X = numpy.array(measures_df).astype(float)

# I want to find ideal number of clusters for the data, too many and there will be no anamolies/overfit
'''
I am using the elbow method to find the ideal number of k, where the elbow hooks will be the ideal k
Anything to the right of that will not return substantial outcomes
The result of running this function shows that ideal number of k will be 4
'''


def find_k(arr):
    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(arr)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker=0)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('ideal_k.png')
    plt.show()


find_k(X)

'''
In the final model, we use 4 clusters, we initialize the centroids at random spots
We will have the algo do 10 runs with no more than 300 iterations for each run to attempt convergence of centroids
The final model will have the best output of those ten runs
We will set a tolerance (tol) to give the model a stopping point in the event it does not converge
'''
km_model = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
results = km_model.fit(X)

'''
I used matplotlib to plot the centroids and see where they landed in relation to the data. 
'''
centroids = km_model.cluster_centers_
labels = km_model.labels_
colors = ['red', 'blue', 'black', 'green']
count = 0

plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', color='black')
plt.scatter(X[:, 0], X[:, 1], marker='o', color='red')
plt.savefig('model.png')
plt.show()

'''
I added the test case from the assignment here and ran it through traditional prediction methods. 
I then plotted the results, which indicated that the point in question was a member of cluster/class number 3
'''
test_case = numpy.array([30, 22]).reshape(1, -1)
pred = km_model.predict(test_case)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', color='black')
plt.scatter(X[:, 0], X[:, 1], marker='o', color='red')
plt.scatter(test_case[:, 0], test_case[:, 1], marker='o', color='blue')
plt.savefig('prediction.png')
plt.show()

'''
There is another prediction method within SKL that is specifically designed for anomaly detection.
This does make an assumption that the data is normally distributed.
The idea is to draw a shape (envelope) around the data and see whether specific data points fall outside of that shape
The shape is drawn using Mahalanobis distance, or the distance between a single point and a distribution
It will return a 1 if the item is an inlier and a -1 if the item is an outlier. 
The result of the prediction was a 1.
'''
km_cov = EllipticEnvelope(random_state=0).fit(X)
pred2 = km_cov.predict(test_case)
print(pred2)

'''
The model does not show the data point 30, 22 as an outlier.
'''