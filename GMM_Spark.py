from pyspark.context import SparkContext
from pyspark.sql import Row, SQLContext
from pyspark.ml.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.ml.linalg import Vectors

Price = {'vhigh': 0, 'high': 1, 'med': 2, 'low' : 3}
Maint = {'vhigh': 0, 'high': 1, 'med': 2, 'low' : 3}
Doors = {'2': 0, '3':1, '4':2, '5more':3}
Person = {'2': 0, '4':1, 'more': 2}
Luggage = {'small':0, 'med':1, 'big':2}
Safety = {'low': 0, 'med':1, 'high':2}
Acceptability = {'unacc':0, 'acc':1, 'vgood':2, 'good':3}

def f(x):
    rel = {}
    rel['features'] = Vectors.dense(Price[str(x[0])],Maint[str(x[1])],Doors[str(x[2])],Person[str(x[3])],Luggage[str(x[4])],Safety[str(x[5])])
    return rel

# Load and parse the data
if __name__=='__main__':
    sc = SparkContext(appName="GMMExample")
    sqlContext = SQLContext(sc)
    df = sc.textFile("file:///usr/local/experiment/car.txt").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    gm = GaussianMixture().setK(4).setPredictionCol("Prediction").setProbabilityCol("Probability")
    gmm = gm.fit(df)
    result = gmm.transform(df)
    result.show(150, False)


