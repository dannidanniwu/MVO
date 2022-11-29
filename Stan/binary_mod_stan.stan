data {
  int<lower=0> N;  //#of observations
  int<lower=0> D; // # of outcome
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=0,upper=1> y[N,D];
  real x[N];       //covariates
}

parameters {
  real z_beta_0;
  real z_beta_1;
  real z_beta_2;
  real z_beta_3;
  vector[N] z_ind_beta_0;
  vector[N] z_ind_beta_1; 
  real<lower=0> sigma_beta_0;
  real<lower=0> sigma_beta_1;
}

transformed parameters {
  vector[D] yhat[N];
  real beta_0;
  real beta_1;
  real beta_2;
  real beta_3;
  vector[N] ind_beta_0;
  vector[N] ind_beta_1;
  //Non-Centered Parameterization
  beta_0 = 10*z_beta_0;
  beta_1 = 10*z_beta_1;
  beta_2 = 10*z_beta_2;
  beta_3 = 10*z_beta_3;
  ind_beta_0 = sigma_beta_0*z_ind_beta_0;
  ind_beta_1 = sigma_beta_1*z_ind_beta_1;
  
  for (i in 1:N)
    for (k in 1:D){
      yhat[i,k] =  beta_0 + beta_1*x[i] + A[i]*(beta_2 + ind_beta_0[i] + x[i]*beta_3 + x[i]* ind_beta_1[i]);
    }
}

model {
  z_beta_0 ~ std_normal(); 
  z_beta_1 ~ std_normal(); 
  z_beta_2 ~ std_normal(); 
  z_beta_3 ~ std_normal(); 
  z_ind_beta_0 ~ std_normal();
  z_ind_beta_1 ~ std_normal();
  for (i in 1:N)
    for (k in 1:D){
       y[i,k] ~ bernoulli_logit(yhat[i,k]);
    }
}

