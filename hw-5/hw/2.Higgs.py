
# coding: utf-8
# Name: Jayant Malani
# Email: jmalani@eng.ucsd.edu
# PID: A53102766
from pyspark import SparkContext
sc = SparkContext()

# In[1]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# ### Higgs data set
# * **URL:** http://archive.ics.uci.edu/ml/datasets/HIGGS#
# * **Abstract:** This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
#
# **Data Set Information:**
# The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.
#
#

# In[2]:

#define feature names
#feature_text='lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
#features=[strip(a) for a in split(feature_text,',')]
#print len(features),features


# ### As done in previous notebook, create RDDs from raw data and build Gradient boosting and Random forests models. Consider doing 1% sampling since the dataset is too big for your local machine

# In[5]:

path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)


# In[11]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.sample(False,0.1, seed=255).map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda x : LabeledPoint(x[0],x[1:])).cache()
#Data.first()


# In[12]:

# count the number of examples of each type
#total=Data.count()
#print 'total data size=',total
#counts=(Data.map(lambda x : (x.label,1)).countByKey()).items()  ## Fillin ##
#print '              type (label):   percent of total'
#print '---------------------------------------------------------'
#print '\n'.join(['(%3.1f):\t%4.2f'%(a[0],100.0*a[1]/float(total)) for a in counts])


# In[14]:

(trainingData,testData)=Data.randomSplit([0.7,0.3], seed=255)
#print 'Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data.count(),trainingData.cache().count(),testData.cache().count())


# In[15]:

## Gradient Boosted Trees ##
from time import time
errors={}
for depth in [10]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData, {},maxDepth=depth, numIterations=40)##FILLIN to generate 10 trees ##)
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda lp: lp.label).zip(Predicted) ### FILLIN ###
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth]#,int(time()-start),'seconds'
#print errors


# In[16]:

#from time import time
#errors={}
#for depth in [20]:
#    start=time()
#    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
#                                     numTrees=10, featureSubsetStrategy="auto",
#                                     impurity='gini', maxDepth=depth, maxBins=32)## FILLIN ##)
#    #print model.toDebugString()
#    errors[depth]={}
#    dataSets={'train':trainingData,'test':testData}
#    for name in dataSets.keys():  # Calculate errors on train and test sets
#        data=dataSets[name]
#        Predicted=model.predict(data.map(lambda x: x.features))
#        LabelsAndPredictions=data.map(lambda lp: lp.label).zip(Predicted) ### FILLIN ###
#        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
#        errors[depth][name]=Err
#    print depth,errors[depth],int(time()-start),'seconds'
#print errors


# In[ ]:



