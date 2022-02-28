functions {
    #include "functions.stan"
    #include "ds.stan"
    #include "dirichlet_alternative.stan"
}


data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations
  
  int<lower=2> k; //number of classes
  int<lower=2> l; //number of labels
  int<lower=1,upper=l> classes[k]; // classes[i] is the label corresponding to the i-th class

  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=l> ann[a]; // the annotation

  vector[k] tau_prior;
  vector[l-1] eta_alpha_prior[k];
  vector[l-1] eta_beta_prior[k];
}

transformed data {
  int dst[k,l-1];
  dst = compute_movements(l, classes);
}

parameters {
  simplex[k] tau;
  vector<lower=0>[l-1] eta[k];
  simplex[l] pi[w,k];
}

transformed parameters {
  simplex[l] pi_h[k];
  pi_h = softmax_diag(eta, classes, dst);
  print("hierarchical_pi", pi_h);

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
  for(k_ in 1:k) {
    eta[k_] ~ gamma(eta_alpha_prior[k_], eta_beta_prior[k_]);
  }

  // Prior over pi given pi_h
  for (w_ in 1:w) {
    for(k_ in 1:k) {
      pi[w_,k_] ~ dirichlet(100*pi_h[k_] + 1);
    }
  }


  // Prior over tau
  tau ~ dirichlet(tau_prior);

  // Observation model

  // Summing over hidden var t_C
  for (t_ in 1:t)
     target += log_sum_exp(log_p_t_C[t_]);

  print("tau:",tau);
  print("eta:",eta);
  print("pi:",pi);
  print("target:",target());
}
