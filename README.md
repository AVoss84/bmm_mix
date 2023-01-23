# *Bernoulli Mixture Models (BMM)*

This repository provides tools in a Python package *bernmix* for the unsupervised analysis of multivariate Bernoulli data with known number of cluster/groups using BMMs. 

## Maximum likelihood estimation 

Shows how to fit the model using [Expectation-Maximizition (EM)](https://github.com/AVoss84/bmm_mix/blob/master/EM_for_BMM.ipynb) algorithm as outlined in *Bishop (2006): Pattern Recognition and Machine Learning*. 

## Fully Bayesian estimation 

Shows how to fit the model using [Gibbs sampling](https://github.com/AVoss84/bmm_mix/blob/master/Gibbs_for_BMM.ipynb) algorithm.

```
from bernmix.utils import bmm_utils as bmm
```

### Installing

```
pip install -r requirements.txt
```

## Authors

* **Alexander Vosseler**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

