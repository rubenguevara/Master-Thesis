# DataPrep
This is a repo for the preparation of data to be used in my master thesis. <br/> 
The data used in this directory can be found in EventSelector. Here we both plot the variables and prepare the data to be read in ML.  <br/> <br/>
The reason we plot the variables is to see if it is in agreement with the datapoints recorded at the LHC. This is to check that the information going into the ML netowrks is physically correct. <br/> <br/>
The data generated there are root ntules and histograms. The root ntuples are made into a Pandas DataFrame in ml_file_maker then made into readable hdf5 files. This code showcases the plots that are used as the variables for machine learning.  <br/> <br/>
The actual .h5 files are not included in this repo due to their size.