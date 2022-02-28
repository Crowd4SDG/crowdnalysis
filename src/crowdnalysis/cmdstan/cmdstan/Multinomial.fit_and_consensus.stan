functions {
    #include "functions.stan"
    #include "multinomial.stan"
}

data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations
  
  int<lower=2> k; // number of classes
  int<lower=2> l; // number of labels
  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=l> ann[a]; // the annotation
  vector[k] tau_prior;
  vector[l] pi_prior[k];
}

parameters {
  simplex[k] tau;
  simplex[l] pi[k];
}

transformed parameters {
  // log_p_t_C[t_][k_] is the log of the probability of the annotations of task t_ assuming t_C=k_
  vector[k] log_p_t_C[t];
  vector[k] t_C[t]; //the true class distribution of each item

  print("a");
  log_p_t_C = multinomial_log_p_t_C(tau, pi, t, t_A, ann);

  // Compute the probabilities from the logs

  for(t_ in 1:t)
    t_C[t_] = softmax(log_p_t_C[t_]);

}


model {

  // Prior over pi

  for(k_ in 1:k)
    pi[k_] ~ dirichlet(pi_prior[k_]);
  
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  // Observation model

  // Summing over hidden var t_C
  for (t_ in 1:t)
  {
     real lse = log_sum_exp(log_p_t_C[t_]);
     target += lse;
  }
}