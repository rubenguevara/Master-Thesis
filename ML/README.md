# ML
Here are the different machine learning algorithms.  <br/>In this project we are studying (Deep, at some later point) Neural Networks using TensorFlow and Boosted Decision Trees using XGBoost. <br/> <br/>
So far the approach is to first train the networks with a file contaning all the SM background and all the different Dark Matter signals into one big file. This is one approach to "teach the network Dark Matter" in a model independent way. <br/> <br/>
The second approach is to train the networks with a file containing all the SM background and only one Dark Matter model. This will then be done for all different available models and the hope is to combine what every network learned from the model it studied into a "smarter network" which has the information of every model. <br/> <br/>
At a later point the network will be used to predict from real data if there are any Dark Matter events recorded. 
