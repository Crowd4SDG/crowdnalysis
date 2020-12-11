# crowdnalysis
 Crowdsourcing Citizen Science projects usually require citizens to classify items (images, pdfs, songs,&#8230;) into one of a finite set of categories. Once an image is classified by different citizens, the different votes need to be aggregated to obtain a consensus classification. Usually this is done by selecting the most voted category. Crowdnalysis allows Crowdsourcing Citizen Science projects to compute consensus that go beyond the selection of the most voted category, by computing a model of quality for each of the citizen scientist involved in the project. Those more advanced consensus result in higher quality information for the Crowdsourcing Citizen Science project.

## Implemented consensus algorithms
- Majority Voting
- Probabilistic
- Dawid-Skene
- Dawid-Skene (`PyStan` version, [see](https://pystan.readthedocs.io/en/latest/))

# How to use as a package
- Clone the repo 
- Add the directory to the `PYTHONPATH`
- `pip install` the requirements into the virtual environment that you will use `crowdnalysis`
- `ìmport crowdnalysis`

*see* `albania-analysis` [repo](https://github.com/Crowd4SDG/albania-analysis) for a sample usage 
and its initialization script (`bin/init-local.sh`) for the above-mentioned configuration. 

## Known issues
If you are working on a 64bit platform and you get an "Architecture not supported" error 
during g++ compilation for `PyStan`,  add the following line into your `~/.bashrc`:

```
# Set architecture to avoid GCC compilation problems with relavant python packages
export ARCHFLAGS="-arch x86_64"

```