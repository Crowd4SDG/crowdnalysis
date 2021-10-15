[![IIIA-CSIC](https://img.shields.io/badge/brewing%20at-IIIA--CSIC-blue)](https://iiia.csic.es)
[![PyPI](https://img.shields.io/pypi/v/crowdnalysis)](https://pypi.org/project/crowdnalysis)
[![codecov](https://codecov.io/gh/Crowd4SDG/crowdnalysis/branch/develop/graph/badge.svg?token=JZ8BD8MZ9D)](https://codecov.io/gh/Crowd4SDG/crowdnalysis)

# crowdnalysis
Crowdsourcing Citizen Science projects usually require citizens to classify items (images, pdfs, songs, etc.) into 
one of a finite set of categories. Once an image is annotated by contributing citizens, we need to aggregate these 
annotations to obtain a consensus classification. Usually, the consensus for an item is achieved by selecting the 
most voted category for the item. *crowdnalysis* allows computing consensus using more advanced techniques beyond the 
standard majority voting. In particular, it provides consensus methods that model quality for each of the citizen 
scientists involved in the project. This more advanced consensus results in higher quality information for the 
Crowdsourcing Citizen Science project, an essential requirement as citizens are increasingly willing and able to 
contribute to science.

## Implemented consensus algorithms

  - Majority Voting
  - Probabilistic
  - Multinomial
  - Dawid-Skene
  
In addition to the pure Python implementations above, the following models are implemented in the 
probabilistic programming language [Stan](https://mc-stan.org) and used via the 
`CmdStanPy` [interface](https://mc-stan.org/cmdstanpy):
  - Multinomial
  - Multinomial Eta
  - Dawid-Skene
  - Dawid-Skene Eta Hierarchical

~ Eta models impose that the probability of a reported label is higher for the real class in the error-rate 
(a.k.a. confusion) matrix.

## Features

  - Import annotation data from a `CSV` file with preprocessing option
  - Set inter-dependencies between questions to filter out irrelevant annotations
  - Distinguish real classes for answers from reported labels (e.g., "Not answered")    
  - Calculate inter-rater reliability with different measures
  - Fit selected model to annotation data and compute the consensus 
  - Compute the consensus with a fixed pre-determined set of parameters
  - Fit the model parameters provided that the consensus is already known
  - Given the parameters of a generative model (Multinomial, Dawid-Skene); sample annotations, tasks, 
  and workers (i.e., annotators)
  - Conduct prospective analysis of the 'accuracy vs. number of annotations' for a given set of models
  - Visualize the error-rate matrix for annotators
  - Visualize the consensus on annotated images in `HTML` format 
  

## Quick start

crowdnalysis is distributed via PyPI: [https://pypi.org/project/crowdnalysis/](https://pypi.org/project/crowdnalysis/)

Install as a standard Python package:

`$ pip install crowdnalysis`

`CmdStanPy` will be installed automatically as a dependency. However, this package requires the installation of the 
`CmdStan` command-line interface too. 
This can be done via executing the `install_cmdstan` utility that comes with `CmdStanPy`.
See related [docs](https://mc-stan.org/cmdstanpy/installation.html) for  more information.

`$ install_cmdstan`

Use the package in code:

`>>> import crowdnalysis`

Check available consensus models:

`>>> crowdnalysis.factory.Factory.list_registered_algorithms()`

See the [tutorial](https://github.com/Crowd4SDG/crowdnalysis/blob/master/nb/tutorial.ipynb) notebook for the usage of 
main features. 

## How to run unit tests

We use [pytest](https://pytest.org) as the testing framework. Tests can be run by:

`$ pytest`

If you want to get the logs of the execution, do 

`$ pytest --log-cli-level 0`

## Logging 

We use the standard `logging` [library](https://docs.python.org/3/howto/logging.html).

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
<img src="https://europa.eu/european-union/sites/europaeu/files/docs/body/flag_yellow_low.jpg" alt="" width="40"/> 
crowdnalysis is being developed within the <a href="https://crowd4sdg.eu/">Crowd4SDG</a> project funded by the 
European Union’s Horizon 2020 research and innovation programme under grant agreement No. 872944. 

## Reference
For the details of the conceptual and mathematical model of crowdnalysis, see: 

[1<a name="ref1"></a>] 
Cerquides, J.; Mülâyim, M.O.; Hernández-González, J.; Ravi Shankar, A.; Fernandez-Marquez, J.L. 
A Conceptual Probabilistic Framework for Annotation Aggregation of Citizen Science Data. *Mathematics* **2021**, 9, 875.
[https://doi.org/10.3390/math9080875](https://doi.org/10.3390/math9080875)