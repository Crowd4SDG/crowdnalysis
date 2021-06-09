data {
  int<lower=1> t; //number of tasks
  int<lower=2> k; //number of classes
}

parameters {
  simplex[k] tau;
}

generated quantities {
  // Generate workers. Nothing to do here
  
  // Generate tasks
  int <lower=1,upper=k> t_C[t];
  
  for (_t in 1:t) {
    t_C[_t] = categorical_rng(tau);
  }
}
