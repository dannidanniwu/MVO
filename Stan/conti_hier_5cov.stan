//compared to version_1
//sigma sd=3
//change the way reparameterization
data {
  int<lower=0> N;  //of observations
  int<lower=0> D; //  of binary outcome
  int<lower=0> P;// of covariates
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  vector[D] y[N]; 
  row_vector[P] x[N];       //covariates
}

parameters {
  real z_beta_0;
  matrix[P,D] z_beta_1;
  matrix[(P+1),D] z_beta_ind[N];
  matrix[(P+1),D] z_beta;
  vector[(P+1)] z_beta_star;
  vector<lower=0>[(P+1)] sigma_beta;
  vector<lower=0>[(P+1)] sigma;
  cholesky_factor_corr[D] L_Phi[(P+1)];
  real<lower=0> tau;
}

transformed parameters {
  vector[D] yhat[N];
  real beta_0;
  matrix[P,D] beta_1;
  matrix[(P+1),D] beta_ind[N];
  matrix[(P+1),D] beta;
  vector[(P+1)] beta_star;
  
  //Non-Centered Parameterization
  beta_0 = 10*z_beta_0;
  beta_star = 10* z_beta_star;
  
  for (j in 1:P){
      beta_1[j] = 10*z_beta_1[j];
   }
   
  
  for (j in 1:(P+1))
     for (k in 1:D){
       beta[j,k] = beta_star[j] + sigma_beta[j]*z_beta[j,k];
  }
   
  for (i in 1:N){
      for(j in 1:(P+1)){
        beta_ind[i,j] = beta[j] + z_beta_ind[i,j]*diag_pre_multiply(rep_vector(sigma[j],D),L_Phi[j]);
       }
         for (k in 1:D){
         yhat[i,k] =  beta_0 + x[i]*col(beta_1,k)  +  append_col(1, x[i]) * col(beta_ind[i],k) * A[i];
    }
  }
}

model {
  
  sigma ~ exponential(3);
  sigma_beta ~ exponential(10);
  for ( j in 1:(P+1)){
   L_Phi[j] ~ lkj_corr_cholesky(1);
  }
  
  tau ~ student_t(3,0,8);
  to_vector(z_beta) ~ std_normal();
  
  for (i in 1:N){
    to_vector(z_beta_ind[i]) ~ std_normal();
    for (k in 1:D){
       y[i,k] ~ normal(yhat[i,k],tau);
    }
  }
}

generated quantities {
  corr_matrix[D] Phi_c[(P+1)];
  for (j in 1:(P+1)){
  Phi_c[j]= L_Phi[j] * L_Phi[j]';
  }
}
