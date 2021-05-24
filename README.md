# DB Scan Clustering Algorithm Jan – April 2021
# Data Mining, CMPT 459, SFU
* Implemented decision tree classification algorithm with error reduction pruning to predict income of an individual given their relevant information such as age, hours per week, native-country etc.
* Algorithm works on both numerical and categorical attributes and uses information gain as attribute split criterion to classify income with 80% accuracy.
* Implemented DB scan clustering algorithm to cluster density-reachable objects and to detect outliers on household electricity usage data. If object was density reachable from two clusters it was assigned to both clusters.
* Standard scaling was implemented to normalize distances between data records. A graph of k-distances was plotted to find the first valley that was used as the epsilon distance provided to the algorithm.
