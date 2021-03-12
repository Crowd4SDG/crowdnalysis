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
  for (_k in 1:k) {
    for (_i in 1:k-1) {
       dst[_k][_i] = _i + (_i>=_k);
    }
    //for (_i in _k:k-1) {
    //   dst[k][_i] = _i+1;
    //}
  }
  print(dst);
}

parameters {
  simplex[k] tau;
  vector<lower=0>[k-1] eta[k];
}

transformed parameters {
  simplex[k] pi[k];
  print("eta",eta);
  for (_k in 1:k) {
    pi[_k][dst[_k]] = -eta[_k];
    pi[_k][_k] = 0.;
    pi[_k] = softmax(to_vector(pi[_k]));
  }
  print("pi",pi);
  // log_p_t_C[_t][_k] is the log of the probability that t_C=_k for task _t 
  vector[k] log_p_t_C[t];
  vector[k] t_C[t]; //the true class distribution of each item

  // Initialize with the prior
  
  log_p_t_C = rep_array(log(tau), t);
  
  // Update log_p_t_C with each of the annotations
  
  { 
        // Make the log and transpose the emission matrix
        vector [k] log_emission_t[k];
        
        log_emission_t = log_transpose(pi);
                
        // Update each task with the information contributed by its annotations 
        
        for (_a in 1:a)
            log_p_t_C[t_A[_a]] += log_emission_t[ann[_a]];
  }

  // Compute the probabilities from the logs

  for(_t in 1:t)
    t_C[_t] = softmax(log_p_t_C[_t]);

}


model {

  // Prior over pi
  for(_k in 1:k) {
    eta[_k] ~ uniform(min_pi_prior[_k], max_pi_prior[_k]);
  }
  
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  // Observation model

  // Summing over hidden var t_C
  for (_t in 1:t)
     target += log_sum_exp(log_p_t_C[_t]);
}
