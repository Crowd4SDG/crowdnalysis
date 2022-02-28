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
  
  for (t_ in 1:t) {
    t_C[t_] = categorical_rng(tau);
  }
}
