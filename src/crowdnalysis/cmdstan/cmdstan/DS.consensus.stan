functions {
    #include "functions.stan"
    #include "ds.stan"
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

  simplex[k] tau;
  simplex[l] pi[w,k];
}

transformed parameters {
  // log_p_t_C[_t][_k] is the log of the probability that t_C=_k for task _t 
  vector[k] log_p_t_C[t];
  vector[k] t_C[t]; //the true class distribution of each item

  log_p_t_C = ds_log_p_t_C(tau, pi, t, t_A, w_A, ann);

  // Compute the probabilities from the logs

  for(_t in 1:t)
    t_C[_t] = softmax(log_p_t_C[_t]);

}


model {

  // Summing over hidden var t_C
  for (_t in 1:t)
     target += log_sum_exp(log_p_t_C[_t]);
}