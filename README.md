# BlazeML
Machine Learning applied to the optimization of the HPX backend of Blaze. This repository allows to generate data using blazemark and allows to fit machine learning models using scikit-learn library and our own custom Decision Tree Classifier.

The repository is structured as follow:

1. Data Generation (contains the bash scripts that are run to generate data files)

1. Data Analysis ( contains python scripts to analyze and vizualize the data generated. Machine learning algorithms are also fit on the Training Set and Evaluated on the Test Set)

1. Benchmarks ( contains python scripts to plot performance graphs for different benchmarks. This allows to compare the old HPX backend and the Machine Learning backend)

1. Models ( contains the header files that represent the classification trees fitted )
