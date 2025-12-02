# MPE_Localization

Hello, here is a short summary of all the files, and what they do. 


Most important File: 
realtime_localization:
    This class trains/imports a knn model based on parameter k.

    First create an instance of the RealtimeLocalizer class, with parameters:
        1) real_f,action: fraction of real data vs synthetic data (maybe just do 1 for this?)
        2) n_neighbors: hyperparameter, that I also call k in my code often. k and n are the same thing. (sorry!)

    Use the function localizer.predict_from_raw_modalities(wifi, light). What this does, is takes in raw (unprocessed) wifi and light data, and returns an (x, y) prediction. It handles all the processing of this data. 

Other files that you probably don't need to look at. These involve tweaking the kNN model, the preprocessing method, or the synthetic data models. 

IMPORTANT: IF you do decide to change the training data, you will have to retrain everything, since the synthetic data model will have been trained for the old data. The process will be this:
    1) Run preprocess_data on the new data
    2) Run gpr_spatial to train the synthetic model based on the new real data
    3) Run gen.py to generate synthetic data based on the new model that can be used for training. 
    4) Run kNN_Model.py to see which hyperparameter k value works best.
    5) Now we can predict data again with realtime_localization.


1) scripts/preprocess_data.py: 

    This file processes the collected data for training. Metadata from pca + standardization/normlization is saved. Outputs are 3 csvs

    1) Loads the wifi + light data
    2) Drops wifi columns with too many nulls (95 percent threshold)
    3) Normalizes wifi (global min/max) and light (per-column min/max)
    4) Standardizes both datasets (zero mean, unit variance)
    5) Applies PCA on wifi features using variance threshold
    6) Saves metadata of PCA (saves the algorithm so that it can be applied to new data.)
    7) Saves cleaned datasets into 3 files: 
        1) wifi_cleaned.csv (cleaned wifi data after PCA)
        2) wifi_pre_pca.csv (cleaned wifi data before PCA, used for training GPR model)
        3) light_cleaned.csv (cleaned light data)

2) synthetic_data_models/gpr_spatial: 
    
    Trains a gpr synthetic model, given: 
        For light: Pocessed data
        For Wifi: Processed data, but BEFORE the PCA model has been ran on it. 

3) synthetic_data_models/gen: 
    1) Loads the GPR models for light and wifi
    2) Generates (x, y) coordinates (default is 2000 pairs of coordinates)
    3) Passes these (x, y) coordinates through the model, returns the wifi + light data
    4) For wifi data, runs the saved pca algorithm on the data. 

4) kNN_Model.py:

    1) Builds a training, validation and test dataset from our processed data. 

    Training set can be given parameter real_frac, ranging from 0-1. Generally I see it perform better on higher percentages of real data, since my synthetic data was not very good! 

    2) We can then run fit_knn_model, passing in the training data (x and y), and parameter n_neighbors.

    The main function does a loop on all splits of data, 1 (100% real) to 0 (100% synthetic), and computes validation MAE for all values of n, from 1-31. 