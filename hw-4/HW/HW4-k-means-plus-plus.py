
# coding: utf-8

# Name: Jayant Malani
# Email: jmalani@eng.ucsd.edu
# PID: A53102766
from pyspark import SparkContext
sc = SparkContext()
# Your program here

# ## K-means++
#
# In this notebook, we are going to implement [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) algorithm with multiple initial sets. The original k-means++ algorithm will just sample one set of initial centroid points and iterate until the result converges. The only difference in this implementation is that we will sample `RUNS` sets of initial centroid points and update them in parallel. The procedure will finish when all centroid sets are converged.

# In[1]:

### Definition of some global parameters.
K = 5  # Number of centroids
RUNS = 25  # Number of K-means runs that are executed in parallel. Equivalently, number of sets of initial points
RANDOM_SEED = 60295531
converge_dist = 0.1 # The K-means algorithm is terminated when the change in the location
                    # of the centroids is smaller than 0.1


# In[2]:

import numpy as np
import pickle
import sys
from numpy.linalg import norm
from matplotlib import pyplot as plt
import pandas

def print_log(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def parse_data(row):
    '''
    Parse each pandas row into a tuple of (station_name, feature_vec),
    where feature_vec is the concatenation of the projection vectors
    of TAVG, TRANGE, and SNWD.
    '''
    return (row[0],
            np.concatenate([row[1], row[2], row[3]]))


def compute_entropy(d):
    '''
    Compute the entropy given the frequency vector `d`
    '''
    d = np.array(d)
    d = 1.0 * d / d.sum()
    return -np.sum(d * np.log2(d))


# Input is list of probability
# output is index
def choice(p):
    '''
    Generates a random sample from [0, len(p)),
    where p[i] is the probability associated with i.
    '''
    random = np.random.random()
    r = 0.0
    for idx in range(len(p)):
        r = r + p[idx]
        if r > random:
            return idx
    assert(False)


def kmeans_init(rdd, K, RUNS, seed):
    '''
    Select `RUNS` sets of initial points for `K`-means++
    '''
    # the `centers` variable is what we want to return
    n_data = rdd.count()
    shape = rdd.take(1)[0][1].shape[0]
    centers = np.zeros((RUNS, K, shape))

    # input - data point, dist, kth point in center
    # output - dist
    def update_dist(vec, dist, k):
        new_dist = norm(vec - centers[:, k], axis=1)**2
        return np.min([dist, new_dist], axis=0)


    # The second element `dist` in the tuple below is the closest distance from
    # each data point to the selected points in the initial set, where `dist[i]`
    # is the closest distance to the points in the i-th initial set.
    dat = rdd.map(lambda p: (p, [np.inf] * RUNS))               .cache()

    # Collect the feature vectors of all data points beforehand, might be
    # useful in the following for-loop
    local_data = rdd.map(lambda (name, vec): vec).collect()

    print len(local_data)

    # Randomly select the first point for every run of k-means++,
    # i.e. randomly select `RUNS` points and add it to the `centers` variable
    sample = [local_data[k] for k in np.random.randint(0, len(local_data), RUNS)]
    centers[:, 0] = sample

    for idx in range(K - 1):
        ##############################################################################
        # Insert your code here:
        ##############################################################################
        # In each iteration, you need to select one point for each set
        # of initial points (so select `RUNS` points in total).
        # For each data point x, let D_i(x) be the distance between x and
        # the nearest center that has already been added to the i-th set.
        # Choose a new data point for i-th set using a weighted probability
        # where point x is chosen with probability proportional to D_i(x)^2
        ##############################################################################

        #data = [ (vector,update_dist(vector,runDist,idx)) for (vector,runDist) in data]
        #runDistVec = np.matrix([y for (x,y) in data])
        #sumDistVec = np.sum(runDistVec, axis = 1)
        #dataProbT = [(x,np.true_divide(y,sumDistVec)) for (x,y) in data]


        dat       = dat.map(lambda (x,y) : (x,update_dist(x[1], y, idx))).cache()
        sumDistVec = dat.map(lambda (x,y) : y)                          .reduce(lambda x,y:x+y)
        dataProbT  = dat.map(lambda (x,y) : (x,y/sumDistVec)).collect()
        dataProb   = np.array([list(y) for (x,y) in dataProbT]).T
        dataProb1   = np.array([x[1] for (x,y) in dataProbT])
        #new_index=[]
        centers[:,idx+1] = [ dataProb1[choice(dataProb[i])] for i in range(RUNS)]
        #for ix in range(RUNS):
        #    new_index.append(choice(dataProb[:,ix]))
        #centers[:,idx+1] = [dataProbT[i][0][1] for i in new_index]
        #centers[:,idx+1] = [dataProb1[i] for i in new_index]
        pass
    return centers


def get_closest(p, centers):
    '''
    Return the indices the nearest centroids of `p`.
    `centers` contains sets of centroids, where `centers[i]` is
    the i-th set of centroids.
    '''
    best = [0] * len(centers)
    closest = [np.inf] * len(centers)
    for idx in range(len(centers)):
        for j in range(len(centers[0])):
            temp_dist = norm(p - centers[idx][j])
            if temp_dist < closest[idx]:
                closest[idx] = temp_dist
                best[idx] = j
    return best


def kmeans(rdd, K, RUNS, converge_dist, seed):
    '''
    Run K-means++ algorithm on `rdd`, where `RUNS` is the number of
    initial sets to use.
    '''
    k_points = kmeans_init(rdd, K, RUNS, seed)
    print_log("Initialized.")
    temp_dist = 1.0

    print "Initialization Completed"
    def shuffle(x):
        l = []
        for i in range(len(x[1])):
            l.append(((i,x[1][i]),x[0]))
        return l

    iters = 0
    st = time.time()
    while temp_dist > converge_dist:
        ##############################################################################
        # INSERT YOUR CODE HERE
        ##############################################################################

        # Update all `RUNS` sets of centroids using standard k-means algorithm
        # Outline:
        #   - For each point x, select its nearest centroid in i-th centroids set
        #   - Average all points that are assigned to the same centroid
        #   - Update the centroid with the average of all points that are assigned to it

        # Insert your code here

        # You can modify this statement as long as `temp_dist` equals to
        # max( sum( l2_norm of the movement of j-th centroid in each centroids set ))
        ##############################################################################

        dataCenter = rdd.map(lambda (station,vector):(vector,get_closest(vector,k_points))).flatMap(shuffle)
        new_points = dataCenter.combineByKey((lambda x: (x, 1)),                                    (lambda acc, value: (np.add(acc[0],value), acc[1]+1)),                                    (lambda acc1, acc2: (np.add(acc1[0],acc2[0]), acc1[1]+acc2[1])))                                .mapValues(lambda value: np.true_divide(value[0],value[1]))                                .collectAsMap()

        temp_dist = np.max([
                np.sum([norm(k_points[idx][j] - new_points[(idx, j)]) for j in range(K)])
                    for idx in range(RUNS)])

        iters = iters + 1
        if iters % 5 == 0:
            print_log("Iteration %d max shift: %.2f (time: %.2f)" %
                      (iters, temp_dist, time.time() - st))
            st = time.time()

        # update old centroids
        # You modify this for-loop to meet your need
        for ((idx, j), p) in new_points.items():
            k_points[idx][j] = p

    return k_points


# In[3]:

## Read data
data = pickle.load(open("../Data/Weather/stations_projections.pickle", "rb"))
rdd = sc.parallelize([parse_data(row[1]) for row in data.iterrows()])
np.random.seed(RANDOM_SEED)
print rdd.take(1)


# In[4]:

# main code

import time

st = time.time()

centroids = kmeans(rdd, K, RUNS, converge_dist, np.random.randint(1000))
group = rdd.mapValues(lambda p: get_closest(p, centroids))            .collect()

print "Time takes to converge:", time.time() - st


# ## Verify your results
# Verify your results by computing the objective function of the k-means clustering problem.

# In[5]:

def get_cost(rdd, centers):
    '''
    Compute the square of l2 norm from each data point in `rdd`
    to the centroids in `centers`
    '''
    def _get_cost(p, centers):
        best = [0] * len(centers)
        closest = [np.inf] * len(centers)
        for idx in range(len(centers)):
            for j in range(len(centers[0])):
                temp_dist = norm(p - centers[idx][j])
                if temp_dist < closest[idx]:
                    closest[idx] = temp_dist
                    best[idx] = j
        return np.array(closest)**2

    cost = rdd.map(lambda (name, v): _get_cost(v, centroids)).collect()
    return np.array(cost).sum(axis=0)

cost = get_cost(rdd, centroids)


# In[6]:

log2 = np.log2

print log2(np.max(cost)), log2(np.min(cost)), log2(np.mean(cost))


# ## Plot the increase of entropy after multiple runs of k-means++

# In[7]:

entropy = []

for i in range(RUNS):
    count = {}
    for g, sig in group:
        _s = ','.join(map(str, sig[:(i + 1)]))
        count[_s] = count.get(_s, 0) + 1
    entropy.append(compute_entropy(count.values()))


# ## Print the final results

# In[9]:

print 'entropy=',entropy
best = np.argmin(cost)
print 'best_centers=',list(centroids[best])


# In[ ]:



