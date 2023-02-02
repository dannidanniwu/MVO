#make sure tbi and y in the same row are for same patients
#add output div
library(simstudy)
library(data.table)
library(cmdstanr)
library(posterior)
library(slurmR)
library(dplyr)
library(ggplot2)
set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/mixed_onelayer_ord_binary_varyintercept.stan");
#mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/mixed_twolayer_hier_varyintercept.stan");
univt_mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/ord_3c_2b_cov.stan");

#mod <- cmdstan_model("./mixed_onelayer_ord_binary_varyintercept.stan");
#univt_mod <- cmdstan_model("./ord_3c_2b_cov.stan");
s_generate <- function(sigma_beta_0=1^2, sigma_beta_1=1^2,
                       sigma_beta_2 = 1^2,sigma_beta_3 = 1^2,
                       sigma_beta_4 = 1^2,sigma_beta_5 = 1^2,
                       basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
                       n_train = 500) {
  #---generate training data---#
  
  def <- defData(varname = "A", formula = "1;1", dist = "trtAssign")
  def <- defData(def, varname="cov_1", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_2", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_3", formula = 0.5, 
                 dist="binary") 
  def <- defData(def, varname="cov_4", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_5", formula = 0.5, 
                 dist="binary") 
  
  def <- defData(def, varname="beta_0_ind",formula = 0, 
                 variance= "..sigma_beta_0", dist="normal") 
  def <- defData(def, varname="beta_1_ind",formula = 0, 
                 variance= "..sigma_beta_1", dist="normal") 
  def <- defData(def, varname="beta_2_ind",formula = 0, 
                 variance= "..sigma_beta_2", dist="normal") 
  def <- defData(def, varname="beta_3_ind",formula = 0, 
                 variance= "..sigma_beta_3", dist="normal")
  def <- defData(def, varname="beta_4_ind",formula = 0, 
                 variance= "..sigma_beta_4", dist="normal") 
  def <- defData(def, varname="beta_5_ind",formula = 0, 
                 variance= "..sigma_beta_5", dist="normal")
  
  #define canonical parameter for 4 outcomes: 
  def <- defData(def, varname = "y_1_hat", 
                 formula = "0.35*cov_1 - 0.4*cov_2 + 0.15*cov_3 + 0.2*cov_4 - 0.21*cov_5 +
                      A*(-0.4 + beta_0_ind + 0.2*cov_1 - 0.1*cov_2 + 0.1*cov_3 + 0.05* cov_4 -0.06*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3+ beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="nonrandom")
  def <- defData(def, varname = "y_2",dist="binary",  
                 formula = "-0.31+ 0.4*cov_1 - 0.38*cov_2 + 0.13*cov_3+ 0.19*cov_4 - 0.22*cov_5 +
                      A*(-0.39 + beta_0_ind + 0.19*cov_1  - 0.11*cov_2 + 0.09*cov_3+ 0.04* cov_4 -0.07*cov_5+ 
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  
  def <- defData(def, varname = "y_3",dist="binary",  
                 formula = "-0.3 + 0.38*cov_1 -0.39*cov_2+ 0.14*cov_3+ 0.18*cov_4 - 0.2*cov_5 +
                      A*(-0.38 + beta_0_ind + 0.18*cov_1  - 0.12*cov_2 + 0.11*cov_3 + 0.06* cov_4 -0.05*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  def <- defData(def, varname = "y_4",dist="binary",  
                 formula = "-0.29 + 0.42*cov_1 -0.41*cov_2+ 0.16*cov_3+ + 0.21*cov_4 - 0.19*cov_5 +
                      A*(-0.41 + beta_0_ind + 0.21*cov_1  - 0.09*cov_2 + 0.12*cov_3 + 0.07* cov_4 -0.04*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  #---Generate data---#
  ds <- genData(n_train, def)
  dd <- genOrdCat(ds,adjVar = "y_1_hat", basestudy, catVar = "y_ord")
  return(dd)
}

mvo_model <- function(generated_data,mod){
  ###--- fit Bayesian hierachical mixed types of model---###
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
  x <- model.matrix( ~ cov_1+cov_2+cov_3 +cov_4+cov_5, data = generated_data)[,-1]
  y_ord <- generated_data$y_ord
  y_b <- generated_data[,c("y_2","y_3","y_4")]
  
  D <- ncol(y_b) 
  P <- ncol(x)
  L <- length(unique(generated_data$y_ord))
  studydata <- list(
    N = nrow(generated_data), y_ord=y_ord, y_b=y_b,
    x=x,A=generated_data$A, D=D, L=L, P=P)
  #J: #outcome
  #D: # of cov
  
  fit <- mod$sample(
    data = studydata,
    refresh = 0,
    chains = 4L,
    parallel_chains = 4L,
    iter_warmup = 500,
    iter_sampling = 2500,
    show_messages = FALSE)
  
  diagnostics_df <- as_draws_df(fit$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##Get posterior draws of all parameters
  draws_dt <- data.frame(as_draws_df(fit$draws()))
  
  
  beta_trt <- draws_dt$beta_int.1.4.
  beta_inter_cov1 <- draws_dt$beta_int.2.4.
  beta_inter_cov2 <- draws_dt$beta_int.3.4.
  beta_inter_cov3 <- draws_dt$beta_int.4.4.
  beta_inter_cov4 <- draws_dt$beta_int.5.4.
  beta_inter_cov5 <- draws_dt$beta_int.6.4.
  
  
  res <- data.table(beta_trt,beta_inter_cov1,
                    beta_inter_cov2,beta_inter_cov3,beta_inter_cov4,beta_inter_cov5,
                    div,tree_hit)
  
  return(res)
}

univt_model <- function(generated_data, univt_mod){
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
  #---single model for odrdinal outcome---###
  x <- model.matrix( ~ cov_1+cov_2+cov_3 +cov_4+cov_5, data = generated_data)[,-1]
  y_ord <- generated_data$y_ord
  P <- ncol(x)
  L <- length(unique(generated_data$y_ord))
  
  studydata <- list(
    N = nrow(generated_data), y_ord=y_ord, 
    x=x,A=generated_data$A,  L=L, P=P)
  
  fit_univt <- univt_mod$sample(
    data = studydata,
    refresh = 0,
    chains = 4L,
    parallel_chains = 4L,
    iter_warmup = 500,
    iter_sampling = 2500,
    show_messages = FALSE)
  
  diagnostics_df <- as_draws_df(fit_univt$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##
  # Get posterior draws of all parameters
  draws_dt <- data.table(as_draws_df(fit_univt$draws()))
  
  
  beta_trt <- draws_dt$`beta_inter[1]`
  beta_inter_cov1 <- draws_dt$`beta_inter[2]`
  beta_inter_cov2 <- draws_dt$`beta_inter[3]`
  beta_inter_cov3 <- draws_dt$`beta_inter[4]`
  beta_inter_cov4 <- draws_dt$`beta_inter[5]`
  beta_inter_cov5 <- draws_dt$`beta_inter[6]`

  
  res_univt <- data.table(
                    beta_trt,beta_inter_cov1,
                    beta_inter_cov2,beta_inter_cov3,beta_inter_cov4,beta_inter_cov5,
                   
                    div,tree_hit)
  #res_m <- round(apply(res,2,mean),3)
  
  return(res_univt)
  
}
s_train <- function(sigma_beta_0=1^2, sigma_beta_1=1^2,
                    sigma_beta_2 = 1^2,sigma_beta_3 = 1^2,
                    sigma_beta_4 = 1^2,sigma_beta_5 = 1^2,
                    basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
                    n_train = 400, mod=mod,univt_mod=univt_mod){
  #---train the model---#
  generated_data <- s_generate(sigma_beta_0=sigma_beta_0, 
                               sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                               sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                               sigma_beta_5 = sigma_beta_5, basestudy= basestudy,
                               n_train = n_train)
  
  #mvo model
  mvo_results <- mvo_model(generated_data,mod)
  univt_results <- univt_model(generated_data,univt_mod)
  
  
  return(list(mvo_results=mvo_results,
                     univt_results =univt_results))
}

opt_mvo_trt <- function(cov,#covariates in the  data
                          coefs,#posterior draws of coefficients 
                          thresh=0){
  ##---Given patients characteristics, derive tbi for each patients using mvo---##
  
  #posterior draws
  beta_trt <- coefs$beta_trt #main effect of trt
  #fixed effect in the interaction term
  beta_inter <- as.matrix(coefs[,c("beta_inter_cov1","beta_inter_cov2","beta_inter_cov3","beta_inter_cov4","beta_inter_cov5")])
  # #return a n_draws*n_test matrix: column[i]: posterior draws of trt effect of individual i: beta_0+x[i]*beta_1
  hte.distr <- apply(cov,1, function(x) beta_trt + beta_inter%*%x)
  prob.tbi.tmp <- apply(hte.distr,2, function(x) x < thresh)
  tbi <- apply(prob.tbi.tmp, 2, mean) #each column: Pr(tbi < 0) for the ith patient
  opt_trt  <- sapply(tbi, function(x) ifelse(x > 0.5, 1, 0)) #if Pr(tbi < 0) greater than 0.5, recommend CCP
  
  return(opt_trt)
}

opt_univt_trt <- function(cov, coefs,thresh = 0){
  ##---Given patients characteristics, derive tbi for each patients using single model for one outcome---##
  
  beta_trt <- coefs$beta_trt #main effect of trt
  #fixed effect in the interaction term
  beta_inter <- as.matrix(coefs[,c("beta_inter_cov1","beta_inter_cov2","beta_inter_cov3","beta_inter_cov4","beta_inter_cov5")])
  
  hte.distr <- apply(cov,1, function(x) beta_trt + beta_inter%*%x)
  prob.tbi.tmp <- apply(hte.distr,2, function(x) x < thresh)
  tbi <- apply(prob.tbi.tmp, 2, mean)
  opt_univt_trt  <- sapply(tbi, function(x) ifelse(x > 0.5, 1, 0)) #if Pr(tbi < 0) greater than 0.5, recommend CCP
  
  return(opt_univt_trt)
}

s_test_generate <- function(sigma_beta_0=1^2, sigma_beta_1=1^2,
                       sigma_beta_2 = 1^2,sigma_beta_3 = 1^2,
                       sigma_beta_4 = 1^2,sigma_beta_5 = 1^2,
                       basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
                       n_test = 2000) {
  #---generate test data and their potential outcomes---#
  
  def <- defData(varname="cov_1", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_2", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_3", formula = 0.5, 
                 dist="binary") 
  def <- defData(def, varname="cov_4", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_5", formula = 0.5, 
                 dist="binary") 
  
  def <- defData(def, varname="beta_0_ind",formula = 0, 
                 variance= "..sigma_beta_0", dist="normal") 
  def <- defData(def, varname="beta_1_ind",formula = 0, 
                 variance= "..sigma_beta_1", dist="normal") 
  def <- defData(def, varname="beta_2_ind",formula = 0, 
                 variance= "..sigma_beta_2", dist="normal") 
  def <- defData(def, varname="beta_3_ind",formula = 0, 
                 variance= "..sigma_beta_3", dist="normal")
  def <- defData(def, varname="beta_4_ind",formula = 0, 
                 variance= "..sigma_beta_4", dist="normal") 
  def <- defData(def, varname="beta_5_ind",formula = 0, 
                 variance= "..sigma_beta_5", dist="normal")
  
  #define canonical parameter for 4 outcomes: 
  def <- defData(def, varname = "y_trt1_hat", 
                 formula = "0.35*cov_1 - 0.4*cov_2 + 0.15*cov_3 + 0.2*cov_4 - 0.21*cov_5 +
                      1*(-0.4 + beta_0_ind + 0.2*cov_1 - 0.1*cov_2 + 0.1*cov_3 + 0.05* cov_4 -0.06*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3+ beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="nonrandom")
  
  def <- defData(def, varname = "y_trt0_hat", 
                 formula = "0.35*cov_1 - 0.4*cov_2 + 0.15*cov_3 + 0.2*cov_4 - 0.21*cov_5",
                 link="nonrandom")
  #the optimsl treatment decision based on potential outcomes
  def2 <- defDataAdd(varname = "optimal_trt",
                     formula = "1*(y_trt0 > y_trt1)",
                     dist="nonrandom")
  
  def2 <- defDataAdd(def2, varname = "y_diff",
                     formula = "y_trt1 - y_trt0",
                     dist="nonrandom")
  #---Generate data---#
  ds <- genData(n_test, def)
  dd <- genOrdCat(ds,adjVar = "y_trt1_hat", basestudy, catVar = "y_ord_trt1")
  dd <- genOrdCat(dd,adjVar = "y_trt0_hat", basestudy, catVar = "y_ord_trt0")
  dd[, y_ord_trt1 := factor(y_ord_trt1, levels = c(1:11))]
  dd[, y_ord_trt0 := factor(y_ord_trt0, levels = c(1:11))]
  dd$y_trt1 <- dd$y_ord_trt1 %>% as.character(.) %>% as.numeric(.)
  dd$y_trt0 <- dd$y_ord_trt0 %>% as.character(.) %>% as.numeric(.)
  dd <- addColumns(def2, dd)
  return(dd)
}

PCD <- function(train_coef,thresh=0,
                sigma_beta_0=sigma_beta_0, 
                sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                sigma_beta_5 = sigma_beta_5,basestudy= basestudy,
                n_test = n_test
                ){
  ##---Proportion of correct decisions on test data---##
  test_data <- s_test_generate(sigma_beta_0=sigma_beta_0, 
                               sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                               sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                               sigma_beta_5 = sigma_beta_5,basestudy= basestudy,
                               n_test = n_test)
  ##Given patients characteristics, derive optimal treatment for each patients usign the true value in data generating process
  cov <- test_data[,c("cov_1","cov_2","cov_3","cov_4","cov_5")]
  
  trt_true <- test_data$optimal_trt
    
  trt_mvo <- opt_mvo_trt(cov=cov, coefs=train_coef$mvo_results, thresh =0)

  trt_univt <- opt_univt_trt(cov=cov, coefs = train_coef$univt_results,thresh = 0)
  
  test_data[ , `:=` (trt_univt= trt_univt,trt_mvo= trt_mvo)]
  
  PCD_by_y <- test_data[,
                        .(pcd_univt = mean(optimal_trt==trt_univt),pcd_mvo = mean(optimal_trt==trt_mvo),.N),
                        keyby = .(y_diff)]
  
  
  return(data.table(PCD_by_y))
  
}

bayes_single_rep <- function(iter, sigma_beta_0=1^2, sigma_beta_1=1^2,
                             sigma_beta_2 = 1^2,sigma_beta_3 = 1^2,
                             sigma_beta_4 = 1^2,sigma_beta_5 = 1^2,
                             basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
                             n_train = 400,
                             n_test=2000,mod=mod,univt_mod=univt_mod,thresh=0) {
  
  train_coef <- s_train(sigma_beta_0=sigma_beta_0, 
                        sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                        sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                        sigma_beta_5 = sigma_beta_5, basestudy= basestudy,
                        n_train = n_train,mod=mod,univt_mod=univt_mod)
  
  test_PCD <- PCD(train_coef=train_coef,thresh=0,
                  sigma_beta_0=sigma_beta_0, 
                  sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                  sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                  sigma_beta_5 = sigma_beta_5,basestudy= basestudy,
                  n_test = n_test) 
  

  div_mvo <- mean(train_coef$mvo_results$div)
  div_univt <- mean(train_coef$univt_results$div)
  return(data.table(iter, test_PCD,
                    div_mvo,
                    div_univt))
                    
}
# bayes_result <- rbindlist(lapply(1, function(x) bayes_single_rep(x,sigma_beta_0=1, sigma_beta_1=1,
#                                                         sigma_beta_2 = 1,sigma_beta_3 = 1,
#                                                         sigma_beta_4 = 1,sigma_beta_5 = 1,
#                                                         basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
#                                                         n_train = 500, n_test=2000,mod=mod,univt_mod=univt_mod)))

# save(bayes_result,file="./bayes_result.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

job <- Slurm_lapply(
  X = 1:200,
  FUN =bayes_single_rep,
  sigma_beta_0=1, sigma_beta_1=1,
  sigma_beta_2 = 1,sigma_beta_3 = 1,
  sigma_beta_4 = 1,sigma_beta_5 = 1,
  basestudy= c(0.100, 0.107, 0.095, 0.085, 0.090, 0.090, 0.108, 0.100, 0.090, 0.075, 0.060),
  n_train = 500, n_test=2000,mod=mod,univt_mod=univt_mod,
  njobs = 60,
  mc.cores = 4L,
  job_name = "mvo_68",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "4:00:00", partition = "cpu_dev", `mem-per-cpu` = "8G"),
  export = c("s_generate", "mvo_model","univt_model","s_train","opt_mvo_trt","opt_univt_trt",
             "s_test_generate","PCD"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/gamma1_n500_PCD_by_actual_y.rda")

####--plot---#####
#check proportion of levels of outcomes
# summary(res[, 97:113])
# bayes_result <- res[res$div_mvo <= 100, ]
# dim(bayes_result)#0.01:71/0.1:41/1:79
#the ordinal outcome
# 
# 
# gener_y4 <- data.frame(iter=1,
# 
#                        beta_cov1_mvo= 0.35,
#                        beta_cov2_mvo = -0.4,
#                        beta_cov3_mvo = 0.15,
#                        beta_cov4_mvo = 0.2,
#                        beta_cov5_mvo = -0.21,
#                        beta_trt_mvo = 0.4,
#                        beta_inter_cov1_mvo=0.2,
#                        beta_inter_cov2_mvo=-0.1,
#                        beta_inter_cov3_mvo=0.1,
#                        beta_inter_cov4_mvo=0.05,
#                        beta_inter_cov5_mvo=-0.06,
#                        beta_star_trt_mvo=0.395,
#                        beta_star_cov1_mvo=0.195,
#                        beta_star_cov2_mvo=-0.105,
#                        beta_star_cov3_mvo=0.105,
#                        beta_star_cov4_mvo= 0.045,
#                        beta_star_cov5_mvo= -0.055,
#                        tau_1_mvo= -0.8,
#                        tau_2_mvo= 0.41,
#                        tau_3_mvo= 1.39,
#                        sigma_beta_cov1_mvo= NA,
#                        sigma_beta_cov2_mvo = NA,
#                        sigma_beta_cov3_mvo = NA,
#                        sigma_beta_cov4_mvo = NA,
#                        sigma_beta_cov5_mvo = NA,
#                        sigma_beta_trt_mvo = NA,
#                        Md="True value")
# M_MVO <- subset(bayes_result,select=c("iter","beta_cov1_mvo","beta_cov2_mvo","beta_cov3_mvo",
#                                       "beta_cov4_mvo","beta_cov5_mvo",
#                                       "beta_trt_mvo","beta_inter_cov1_mvo",
#                                       "beta_inter_cov2_mvo","beta_inter_cov3_mvo","beta_inter_cov4_mvo","beta_inter_cov5_mvo",
#                                       "tau_1_mvo","tau_2_mvo","tau_3_mvo","beta_star_trt_mvo",
#                                       "beta_star_cov1_mvo",
#                                       "beta_star_cov2_mvo","beta_star_cov3_mvo","beta_star_cov4_mvo","beta_star_cov5_mvo",
#                                       "sigma_beta_cov1_mvo","sigma_beta_cov2_mvo","sigma_beta_cov3_mvo",
#                                       "sigma_beta_cov4_mvo","sigma_beta_cov5_mvo","sigma_beta_trt_mvo" ))
# M_MVO$Md <- "MVO"
# 
# M_UNI <- subset(bayes_result,select=c("iter","beta_cov1_univt","beta_cov2_univt","beta_cov3_univt",
#                                       "beta_cov4_univt","beta_cov5_univt",
#                                       "beta_trt_univt","beta_inter_cov1_univt",
#                                       "beta_inter_cov2_univt","beta_inter_cov3_univt","beta_inter_cov4_univt","beta_inter_cov5_univt",
#                                       "tau_1_univt","tau_2_univt","tau_3_univt"))%>%mutate("beta_star_trt_univt"=NA,
#                                       "beta_star_cov1_univt"=NA,
#                                      "beta_star_cov2_univt"=NA,
#                                      "beta_star_cov3_univt"=NA,"beta_star_cov4_univt"=NA,"beta_star_cov5_univt"=NA,
# 
#                                       "sigma_beta_cov1_univt"=NA, "sigma_beta_cov2_univt"=NA,
#                                       "sigma_beta_cov3_univt"=NA, "sigma_beta_cov4_univt"=NA,
#                                       "sigma_beta_cov5_univt"=NA, "sigma_beta_trt_univt"=NA,"Md"="UNIO")
# colnames(M_UNI) <-c("iter","beta_cov1_mvo","beta_cov2_mvo","beta_cov3_mvo",
#                      "beta_cov4_mvo","beta_cov5_mvo",
#                      "beta_trt_mvo","beta_inter_cov1_mvo",
#                      "beta_inter_cov2_mvo","beta_inter_cov3_mvo","beta_inter_cov4_mvo","beta_inter_cov5_mvo",
#                      "tau_1_mvo","tau_2_mvo","tau_3_mvo","beta_star_trt_mvo",
#                      "beta_star_cov1_mvo",
#                      "beta_star_cov2_mvo","beta_star_cov3_mvo","beta_star_cov4_mvo","beta_star_cov5_mvo",
#                      "sigma_beta_cov1_mvo","sigma_beta_cov2_mvo","sigma_beta_cov3_mvo",
#                      "sigma_beta_cov4_mvo","sigma_beta_cov5_mvo", "sigma_beta_trt_mvo",
#                     "Md")
# 
# D_all <-rbind(M_UNI,M_MVO,gener_y4)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="Posterior mean of paramters (no individual beta's layer, outcome specific tau)",
#        y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
# 
# 
# 
# # ##########sd#############
# SD_MVO <- subset(bayes_result,select=c("iter","beta_cov1_sd_mvo","beta_cov2_sd_mvo","beta_cov3_sd_mvo",
#                                       "beta_cov4_sd_mvo","beta_cov5_sd_mvo",
#                                       "beta_trt_sd_mvo","beta_inter_cov1_sd_mvo",
#                                       "beta_inter_cov2_sd_mvo","beta_inter_cov3_sd_mvo","beta_inter_cov4_sd_mvo","beta_inter_cov5_sd_mvo",
#                                       "tau_1_sd_mvo","tau_2_sd_mvo","tau_3_sd_mvo","beta_star_trt_sd_mvo",
#                                       "beta_star_cov1_sd_mvo",
#                                       "beta_star_cov2_sd_mvo","beta_star_cov3_sd_mvo","beta_star_cov4_sd_mvo","beta_star_cov5_sd_mvo"))
# SD_MVO$Md <- "MVO"
# 
# SD_UNI <- subset(bayes_result,select=c("iter","beta_cov1_sd_univt","beta_cov2_sd_univt","beta_cov3_sd_univt",
#                                       "beta_cov4_sd_univt","beta_cov5_sd_univt",
#                                       "beta_trt_sd_univt","beta_inter_cov1_sd_univt",
#                                       "beta_inter_cov2_sd_univt","beta_inter_cov3_sd_univt","beta_inter_cov4_sd_univt","beta_inter_cov5_sd_univt",
#                                       "tau_1_sd_univt","tau_2_sd_univt","tau_3_sd_univt"))%>%mutate("beta_star_trt_sd_univt"=NA,
#                                                                                            "beta_star_cov1_sd_univt"=NA,
#                                                                                            "beta_star_cov2_sd_univt"=NA,
#                                                                                            "beta_star_cov3_sd_univt"=NA,"beta_star_cov4_sd_univt"=NA,"beta_star_cov5_sd_univt"=NA,"Md"="UNIO")
# colnames(SD_UNI) <-c("iter","beta_cov1_sd_mvo","beta_cov2_sd_mvo","beta_cov3_sd_mvo",
#                      "beta_cov4_sd_mvo","beta_cov5_sd_mvo",
#                      "beta_trt_sd_mvo","beta_inter_cov1_sd_mvo",
#                      "beta_inter_cov2_sd_mvo","beta_inter_cov3_sd_mvo","beta_inter_cov4_sd_mvo","beta_inter_cov5_sd_mvo",
#                      "tau_1_sd_mvo","tau_2_sd_mvo","tau_3_sd_mvo","beta_star_trt_sd_mvo",
#                      "beta_star_cov1_sd_mvo",
#                      "beta_star_cov2_sd_mvo","beta_star_cov3_sd_mvo","beta_star_cov4_sd_mvo","beta_star_cov5_sd_mvo",
#                     "Md")
# 
# D_all <-rbind(SD_UNI,SD_MVO)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="Posterior SD of paramters (no individual beta's layer, outcome specific tau)",
#        y = "Posterior SD")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
# 
# 
# #####PCD#########
# PCD<- bayes_result[,c("iter","mvo_PCD","univt_PCD")]
# 
# m_data <- reshape2::melt(PCD,id=c("iter"))
# 
# ggplot(m_data, aes(x=variable, y=value)) +
#   geom_boxplot()+
#   labs(title="PCD based on simualtions with low div (no individual beta's layer, outcome specific tau)",
#        y = "PCD", x= "model")+
#   labs(fill= "Model")
# 
# 
# #####PCD#########
# PCD$diff <- PCD$mvo_PCD -PCD$univt_PCD
# round(summary(PCD$diff),3)
# ggplot(PCD, aes(y=diff)) +
#   geom_boxplot()+
#   labs(title="PCD difference based on simualtions with low div (no individual beta's layer, outcome specific tau)",
#        y = "PCD difference")
# # 

# bayes_result <- res[res$div_univt <= 100, ]
# dim(bayes_result)
# #bayes_result$y_diff_abs <- 0:10
# #PCD<- bayes_result[,c("iter","y_diff_abs","mvo_PCD","univt_PCD")]
# 
# PCD<- bayes_result[,c("iter","y_diff","pcd_by_ydiff","pcd_by_or")]
# 
# 
# m_data <- reshape2::melt(PCD,id=c("iter","y_diff"))
# m_data$y_diff <- as.factor(m_data$y_diff)
# ggplot(m_data, aes(x=y_diff, y=value,fill= variable)) +
#   geom_boxplot()+
#   labs(title="PCD grouped by difference in value of potential outcomes (n_train=500)",
#        y = "PCD", x= "difference in potential outcomes")+
#   labs(fill= "Criteria")
# 
# bayes_result[,.(univt_PCD = round(mean(pcd_univt),3),mvo_PCD = round(mean(pcd_mvo),3), N= round(mean(N),0)),
#              keyby = .(y_diff_abs)]