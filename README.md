# Variational Combinatorial Sequential Monte Carlo Methods for Bayesian Phylogenetic Inference

This code provides a reference implementation of the Variational Combinatorial Sequential Monte Carlo algorithms described in the publications: 

* [Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetic Inference](http://www.cs.columbia.edu/~amoretti/papers/phylo.pdf). \
  Moretti, A.\, Zhang, L., Pe'er, I. \
  Machine Learning in Computational Biology, 2020.

* [Variational Combinatorial Sequential Monte Carlo Methods for Bayesian Phylogenetic Inference](). \
  Moretti, A.\*, Zhang, L.\*, Naesseth, C., Venner, H., Blei, D., Pe'er, I. \
  Under Review, 2021

<img src="https://github.com/amoretti86/phylo/blob/master/data/figures/primatesTVCSMC_5.png"
     alt="VCSMC Figure"
     style="float: left; margin-right: 10px;" />


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
