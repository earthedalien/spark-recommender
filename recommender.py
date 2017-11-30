from pyspark.mllib.recommendation import ALS,MatrixFactorizationModel, Rating
from pyspark import SparkContext

sc = SparkContext ()

#Replace filepath with appropriate data
movielens = sc.textFile("/home/foo/Desktop/iDevji/recommender/u.data")

movielens.first() #u'196\t242\t3\t881250949'
movielens.count() #100000

#Clean up the data by splitting it, 
#movielens readme says the data is split by tabs and
#is user product rating timestamp
clean_data = movielens.map(lambda x:x.split('\t'))

#We'll need to map the movielens data to a Ratings object 
#A Ratings object is made up of (user, item, rating)
mls = movielens.map(lambda l: l.split('\t'))
ratings = mls.map(lambda x: Rating(int(x[0]),\
    int(x[1]), float(x[2])))


#Setting up the parameters for ALS
rank = 5 # Latent Factors to be made
numIterations = 10 # Times to repeat process

#Need a training and test set, test set is not used in this example.
train, test = ratings.randomSplit([0.7,0.3],7856)

#Create the model on the training data
model = ALS.train(train, rank, numIterations)

# For Product X, Find N Users to Sell To
model.recommendUsers(242,100)

# For User Y Find N Products to Promote
model.recommendProducts(196,10)

#Predict Single Product for Single User
model.predict(196, 242)

# Predict Multi Users and Multi Products
# Pre-Processing
pred_input = train.map(lambda x:(x[0],x[1]))
