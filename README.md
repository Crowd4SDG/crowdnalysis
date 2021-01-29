[![IIIA-CSIC](https://img.shields.io/badge/brewing%20at-IIIA--CSIC-blue)](https://iiia.csic.es)

# crowdnalysis
 Crowdsourcing Citizen Science projects usually require citizens to classify items (images, pdfs, songs,&#8230;) 
 into one of a finite set of categories. Once an image is classified by different citizens, the different votes 
 need to be aggregated to obtain a consensus classification. Usually this is done by selecting the most voted category. 
 *Crowdnalysis* allows Crowdsourcing Citizen Science projects to compute consensus that go beyond the selection of 
 the most voted category, by computing a model of quality for each of the citizen scientist involved in the project. 
 This more advanced consensus results in higher quality information for the Crowdsourcing Citizen Science project.

## Implemented consensus algorithms
- Majority Voting
- Probabilistic
- Dawid-Skene
- Dawid-Skene (`PyStan` version, [see](https://pystan.readthedocs.io/en/latest/))

## How to use as a package
- Clone the repo 
- Add the `src` directory to `PYTHONPATH`
- `pip install` the requirements into the virtual environment where you will use `crowdnalysis`
- `import crowdnalysis`

*see* `albania-analysis` [repo](https://github.com/Crowd4SDG/albania-analysis) for a sample usage 
and its initialization script (`bin/init-local.sh`) for the above-mentioned configuration. 

## Known issues
### On Mac
- If you are working on a 64-bit Intel platform, and you get an "Architecture not supported" error 
during GCC c++ compilation for `PyStan`, add the following line in your `~/.bashrc`:

```
# Set architecture to avoid GCC compilation problems with relavant python packages
export ARCHFLAGS="-arch x86_64"
```

- If `pystan.StanModel.sampling` method seems to run indefinitely, re-run it with the `verbose=True` parameter. 
If you see the error `ModuleNotFoundError: No module named 'stanfit4anon_model...`, 
insert the following lines on top of your imports 
([hat-tip](https://discourse.mc-stan.org/t/pystan-throws-error-when-running-chains-in-parallel-n-jobs-1/17563/4)):
  
```
import multiprocessing
multiprocessing.set_start_method("fork")
```
