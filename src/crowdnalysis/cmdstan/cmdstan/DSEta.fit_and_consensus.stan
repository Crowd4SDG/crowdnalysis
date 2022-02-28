functions {
    #include "functions.stan"
}


data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations
  
  int<lower=2> k; //number of classes
  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=k> ann[a]; // the annotation
  vector[k] tau_prior;
  vector[k-1] min_pi_prior[k];
  vector[k-1] max_pi_prior[k];
}

transformed data {
  int dst[k,k-1];
  dst = compute_movements(k);
}

parameters {
  simplex[k] tau;
  vector<lower=0>[k-1] eta[w,k];
}

transformed parameters {
  vector[k] pi[w,k];

  for (w_ in 1:w)
    pi[w_] = softmax_diag(eta[w_], dst);

  // print("pi",pi);
  // log_p_t_C[t_][k_] is the log of the probability that t_C=k_ for task t_
  vector[k] log_p_t_C[t];
  log_p_t_C = ds_log_p_t_C(tau, pi, t, t_A, w_A, ann);

  // Compute the probabilities from the logs (maybe this should move to generated quantities)
  vector[k] t_C[t]; //the true class distribution of each item
  for(t_ in 1:t)
    t_C[t_] = softmax(log_p_t_C[t_]);
}


model {

  // Prior over eta
  // for (w_ in 1:w) {
  //   for(k_ in 1:k) {
  //     eta[w_,k_] ~ uniform(min_pi_prior[k_], max_pi_prior[k_]);
  //   }
  // }
  
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  // Observation model

  // Summing over hidden var t_C
  for (t_ in 1:t)
     target += log_sum_exp(log_p_t_C[t_]);
}
