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
  vector[k] t_C[t];
}

transformed data {
  int dst[k,l-1];
  dst = compute_movements(l, classes);
  vector[k] sum_t_C = rep_vector(0,k);

  for (_t in 1:t)
    sum_t_C += t_C[_t];
}

parameters {
  simplex[k] tau;
  vector<lower=0>[l-1] eta[k];
  simplex[l] pi[w,k];
}

transformed parameters {
  simplex[l] pi_h[k];
  pi_h = softmax_diag(eta, classes, dst);
}


model {
  // Prior over eta
  for(_k in 1:k) {
    eta[_k] ~ gamma(eta_alpha_prior[_k], eta_beta_prior[_k]);
  }

  // Prior over pi given pi_h
  for (_w in 1:w) {
    for(_k in 1:k) {
      pi[_w,_k] ~ dirichlet(pi_h[_k] + 1);
    }
  }
  
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  target += dot_product(sum_t_C, log(tau));
  {
        // Make the log and transpose the emission matrix
        vector[k] log_emission_t[w,l];

        log_emission_t = ds_log_transpose(pi);

        // Probability of each annotation

        for (_a in 1:a)
          target += dot_product(log_emission_t[w_A[_a],ann[_a]] , t_C[t_A[_a]]);
  }
}
