data {
  int<lower=0> N;  //#of observations
  int<lower=0> D; // # of outcome
  int<lower=2> L;
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=0,upper=L> y[N,D];
  real x[N];       //covariates
}

parameters {
  
  real z_beta_1;
  real z_beta_2;
  real z_beta_3;
  ordered[L-1] tau;
  vector[N] z_ind_beta_0;
  vector[N] z_ind_beta_1; 
  real<lower=0> sigma_beta_0;
  real<lower=0> sigma_beta_1;
}

transformed parameters {
  vector[D] yhat[N];

  real beta_1;
  real beta_2;
  real beta_3;
  vector[N] ind_beta_0;
  vector[N] ind_beta_1;
  //Non-Centered Parameterization
 
  beta_1 = 10*z_beta_1;
  beta_2 = 10*z_beta_2;
  beta_3 = 10*z_beta_3;
  ind_beta_0 = sigma_beta_0*z_ind_beta_0;
  ind_beta_1 = sigma_beta_1*z_ind_beta_1;
  
  for (i in 1:N)
    for (k in 1:D){
      yhat[i,k] =  beta_1*x[i] + A[i]*(beta_2 + ind_beta_0[i] + x[i]*beta_3 + x[i]* ind_beta_1[i]);
    }
}

model {
  
  z_beta_1 ~ std_normal(); 
  z_beta_2 ~ std_normal(); 
  z_beta_3 ~ std_normal(); 
  z_ind_beta_0 ~ std_normal();
  z_ind_beta_1 ~ std_normal();
  sigma_beta_0 ~ exponential(1);
  sigma_beta_1 ~ exponential(1);
  
  for (l in 1: (L-1)){
    tau[l] ~ student_t(3,0,8);
  }
  for (i in 1:N)
    for (k in 1:D){
       y[i,k] ~ ordered_logistic(yhat[i,k],tau);
    }
}

