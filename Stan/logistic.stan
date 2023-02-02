//compared to version_1
//sigma sd=3
//change the way reparameterization
data {
  int<lower=0> N;  //of observations
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=0,upper=1> y[N];
  real x[N];        //covariates
}

parameters {
  real z_beta_1;
  real z_beta_trt;
  real z_beta_inter;
  real tau;
}

transformed parameters {
  real yhat[N];
  real beta_1;
  real beta_trt;
  real beta_inter;
  
  //Non-Centered Parameterization
  beta_1 = 10*z_beta_1;
  beta_inter = 10*z_beta_inter;
  beta_trt = 10*z_beta_trt;
   
  for (i in 1:N){
        yhat[i] = tau + x[i]*beta_1  +   beta_trt*A[i] + beta_inter * x[i] * A[i];
    }
}

model {


  tau ~ student_t(3,0,8);
  
  
  z_beta_inter ~ std_normal(); 
  z_beta_1 ~ std_normal(); 
  z_beta_trt ~ std_normal(); 
  
  for (i in 1:N){
    y[i] ~ bernoulli_logit(yhat[i]);
  }
}


