# Variational Combinatorial Sequential Monte Carlo Methods for Bayesian Phylogenetic Inference

This code provides a reference implementation of the Variational Combinatorial Sequential Monte Carlo algorithms described in the publications: 

* [Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetic Inference](http://www.cs.columbia.edu/~amoretti/papers/phylo.pdf). \
  Moretti, A.\, Zhang, L., Pe'er, I. \
  Machine Learning in Computational Biology, 2020.

* [Variational Combinatorial Sequential Monte Carlo Methods for Bayesian Phylogenetic Inference](). \
  Moretti, A.\*, Zhang, L.\*, Naesseth, C., Venner, H., Blei, D., Pe'er, I. \
  Under Review, 2021
  
VCSMC builds upon the Combinatorial Sequential Monte Carlo method which is also implemented as a reference:

* [Bayesian Phylogenetic Inference Using a Combinatorial Sequential Monte Carlo Method(https://www.stats.ox.ac.uk/~doucet/wang_bouchardcote_doucet_BayesianphylogeneticscombinatorialSMC_JASA2015.pdf). \ 
Liangliang Wang, Alexandre Bouchard-Côté & Arnaud Doucet (2015). \
Journal of the American Statistical Association, 110:512, 1362-1374, DOI: 10.1080/01621459.2015.1054487


## Usage
To run, type the folowing in terminal: 

`python runner.py 
   --dataset=[some_data] 
   --n_particles=[some_number]
   --batch_size=[some_number]
   --learning_rate=[some_number]
   --twisting=[true/false]
   --jcmodel=[true/false]
   --num_epoch=100`   

This runner.py file assumes that all datasets (`primate.p`, for example) are directly put under a folder called 'data'

<img src="https://github.com/amoretti86/phylo/blob/master/data/figures/primatesTVCSMC_5.png"
     alt="VCSMC Figure"
     style="float: left; margin-right: 10px;" />
     
*Log likelihood values for VCSMC (blue) with K = {4,8,16,32,64,128} samples and VNCSMC (red) with K = {4,8,16,32,64,128} and M = 1 samples on the primates data averaged across 5 random seeds. Higher values of K
produce tighter ELBO / larger log likelihood values with lower stochastic gadient noise.*

