functions {
    #include "functions.stan"
    #include "multinomial.stan"
}

data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations
  
  int<lower=2> k; //number of classes
  int<lower=2> l; //number of labels
  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=l> ann[a]; // the annotation
  vector[k] tau_prior;
  vector[l] pi_prior[k];
  vector[k] t_C[t];
}

transformed data {
  vector[k] sum_t_C = rep_vector(0,k);

  for (t_ in 1:t)
    sum_t_C += t_C[t_];

}

parameters {
  simplex[k] tau;
  simplex[l] pi[k];
}

model{

  // Prior over pi
  for(k_ in 1:k)
    pi[k_] ~ dirichlet(pi_prior[k_]);
  
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  target += dot_product(sum_t_C, log(tau));
  {
        // Make the log and transpose the emission matrix
        vector[k] log_emission_t[l];

        log_emission_t = log_transpose(pi);

        // Probability of each annotation

        for (a_ in 1:a)
          target += dot_product(log_emission_t[ann[a_]] , t_C[t_A[a_]]);
  }
}