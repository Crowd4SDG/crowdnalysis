
"""
this python file compiles a stan file and saves it as a pickle document. Later on you can call this pickle document
and operate with the corresponding stan file without compiling it every time.
"""


import importlib.resources
import pickle
import pystan

with importlib.resources.path("crowdnalysis", "DS_stan_fit.stan") as path_: #this is the path where the satn file lives
    DS_STAN_PATH = str(path_)


sm = pystan.StanModel(file = DS_STAN_PATH) #in order to save the pickle file we must compile the stan file one time

with open('DS_stan_fit.pkl', 'wb') as f: #this is the name of the saved pickle file.
    pickle.dump(sm, f)


print("DS.stan saved successfully :)")