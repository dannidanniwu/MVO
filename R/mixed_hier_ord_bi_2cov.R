#no intercept beta_0 since beta_0 and tau may not be identifiable
#mixed types of outcomes
library(simstudy)
library(data.table)
library(cmdstanr)
library(posterior)
library(slurmR)
library(dplyr)
library(ggplot2)
set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/mixed_hier_ord_binary_2cov.stan");


#mod <- cmdstan_model("./mixed_hier_ord_binary_2cov.stan");
s_generate <- function(sigma_beta_0=0.2, sigma_beta_1=0.1,
                       sigma_beta_2 = 0.1,
                       basestudy= c(.31, .29, .20, .20),
                       n_train = 400) {
  
  def <- defData(varname = "A", formula = "1;1", dist = "trtAssign")
  def <- defData(def, varname="cov_1", formula = "0", 
                 variance= 1, dist="normal") 
  def <- defData(def, varname="cov_2", formula = "0", 
                 variance= 1, dist="normal") 
  
  def <- defData(def, varname="beta_0_ind",formula = "0", 
                 variance= "..sigma_beta_0", dist="normal") 
  def <- defData(def, varname="beta_1_ind",formula = "0", 
                 variance= "..sigma_beta_1", dist="normal") 
  def <- defData(def, varname="beta_2_ind",formula = "0", 
                 variance= "..sigma_beta_2", dist="normal") 
  
  #define canonical parameter for 4 outcomes: 
  def <- defData(def, varname = "y_1_hat", 
                 formula = "0.35*cov_1 - 0.4*cov_2+
                      A*(0.4 + beta_0_ind + 0.2*cov_1 - 0.1*cov_2 + beta_1_ind*cov_1 + beta_2_ind*cov_2)",
                 link="nonrandom")
  def <- defData(def, varname = "y_2",dist="binary",  
                 formula = "-0.3+ 0.4*cov_1 - 0.38*cov_2+
                      A*(0.39 + beta_0_ind + 0.19*cov_1  - 0.11*cov_2 + beta_1_ind*cov_1 + beta_2_ind*cov_2)",
                 link="logit")
  
  def <- defData(def, varname = "y_3",dist="binary",  
                 formula = "-0.3 + 0.38*cov_1 -0.39*cov_2+
                      A*(0.38 + beta_0_ind + 0.18*cov_1  - 0.12*cov_2 + beta_1_ind*cov_1 + beta_2_ind*cov_2)",
                 link="logit")
  def <- defData(def, varname = "y_4",dist="binary",  
                 formula = "-0.3 + 0.42*cov_1 -0.41*cov_2+
                      A*(0.41 + beta_0_ind + 0.21*cov_1  - 0.09*cov_2 + beta_1_ind*cov_1 + beta_2_ind*cov_2)",
                 link="logit")
  #---Generate data---#
  ds <- genData(n_train, def)
  dd <- genOrdCat(ds,adjVar = "y_1_hat", basestudy, catVar = "y_ord")
}

mvo_model <- function(iter, generated_data,mod){
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
  x <- model.matrix( ~ cov_1+cov_2, data = generated_data)[,-1]
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
    show_messages = FALSE,
    adapt_delta = 0.85)
  
  diagnostics_df <- as_draws_df(fit$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##Get posterior draws of all parameters
  draws_dt <- data.frame(as_draws_df(fit$draws()))
  
  
  #fixed effect in the interaction term for the first outcome
  beta_0 <- draws_dt$beta_0
  beta_cov1 <- draws_dt$beta_1.1.4#covariate main effect
  beta_cov2 <- draws_dt$beta_1.2.4
  
  beta_trt <- draws_dt$beta.1.4.
  beta_inter_cov1 <- draws_dt$beta.2.4.
  beta_inter_cov2 <- draws_dt$beta.3.4.
  
  tau_1 <- draws_dt$tau.1.
  tau_2 <- draws_dt$tau.2.
  tau_3 <- draws_dt$tau.3.
  
  Phi_1 <- draws_dt[,c(grep("^Phi_c.1",colnames(draws_dt),value=TRUE))]
  Phi_2 <- draws_dt[,c(grep("^Phi_c.2",colnames(draws_dt),value=TRUE))]
  Phi_3 <- draws_dt[,c(grep("^Phi_c.3",colnames(draws_dt),value=TRUE))]
  #overall mean of betas: beta^*
  beta_star_1 <- draws_dt[,c(grep("^beta_star.1",colnames(draws_dt),value=TRUE))]
  beta_star_2 <- draws_dt[,c(grep("^beta_star.2",colnames(draws_dt),value=TRUE))]
  beta_star_3 <- draws_dt[,c(grep("^beta_star.3",colnames(draws_dt),value=TRUE))]
  
  res <- data.table(beta_0,beta_cov1,beta_cov2,beta_trt,beta_inter_cov1,
                    beta_inter_cov2,
                    tau_1,tau_2,tau_3,Phi_1,Phi_2,Phi_3,beta_star_1,beta_star_2, 
                    beta_star_3,
                    div,tree_hit)
  res_m <- round(apply(res,2,mean),3)
  
  return(data.table(cbind(iter,t(res_m))))
}
s_train <- function(iter, sigma_beta_0=0.2, sigma_beta_1=0.1,
                    sigma_beta_2 = 0.1,
                    basestudy= c(.31, .29, .20, .20),
                    n_train = 400, mod=mod){
  generated_data <- s_generate(basestudy= basestudy,sigma_beta_0=sigma_beta_0, 
                               sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                               n_train = n_train)
  
  #mvo model
  mvo_results <- mvo_model(iter=iter,generated_data,mod)
  return(mvo_results)
}
# bayes_result <- rbindlist(lapply(1, function(x) s_train(x,beta_0=-0.7, beta_1_1=0.35,
#                                                         beta_1_2=0.4,beta_1_3=0.38,
#                                                         beta_1_4=0.42,
#                                                         beta_2=0.4, 
#                                                         beta_3=0.2,#controls the contribution from the two sources of variation. 
#                                                         sigma_beta_0=0.2, sigma_beta_1=0.1, 
#                                                         sigma_beta_2 = 0.01,sigma_beta_3 = 0.025,
#                                                         basestudy= c(.31, .29, .20, .20),
#                                                         n_train = 400,mod=mod)))

# save(bayes_result,file="./bayes_result.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

job <- Slurm_lapply(
  X = 1:100,
  FUN = s_train,
  sigma_beta_0=0.2, sigma_beta_1=0.1,
  sigma_beta_2 = 0.1,
  basestudy= c(.31, .29, .20, .20),
  n_train = 400, mod=mod,
  njobs = 50,
  mc.cores = 4L,
  job_name = "mvo_16",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "8:00:00", partition = "cpu_short", `mem-per-cpu` = "5G"),
  export = c("s_generate", "mvo_model"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/mixed_hier_ord_bi_2cov.rda")




####--plot---#####
# bayes_result <- res[res$div <=100,]#92
# #the ordinal outcome
# gener_y4 <- data.frame(iter=1,
#                        beta_0= -0.3,
#                        beta_cov1= 0.35,
#                        beta_cov2 = -0.4,
#                        beta_trt = 0.4,
#                        beta_inter_cov1=0.2,
#                        beta_inter_cov2=-0.1,
#                        beta_star_1=0.395,
#                        beta_star_2=0.195, 
#                        beta_star_3=-0.105,
#                        tau_1= -0.8,
#                        tau_2= 0.41,
#                        tau_3= 1.39,
#                        Md="True value")
# M_MVO <- subset(bayes_result,select=c(iter,beta_0,beta_cov1,beta_cov2,beta_trt,beta_inter_cov1,
#                                       beta_inter_cov2,
#                                       tau_1,tau_2,tau_3,beta_star_1,beta_star_2, 
#                                       beta_star_3))
# M_MVO$Md <- "Estimated"
# D_all <-rbind(M_MVO,gener_y4)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="MVO: Posterior mean of paramters",
#        y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")

# ###---Check data generation---#
# odds <- function(x){
#   y = 1/(1-x)
#   return(y)
# }
# logOdds.upexp <- log(odds(cumsum(dd[A==0,prop.table(table(y_1))])))
# 
# library(ordinal)
# clmFit <- clmm2(y_1 ~ cov_1 + A + A*cov_1,data=dd)
# summary(clmFit)
