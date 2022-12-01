#no intercept beta_0 since beta_0 and tau may not be identifiable
library(simstudy)
library(data.table)
library(cmdstanr)
library(posterior)
library(slurmR)
library(dplyr)
library(ggplot2)
set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/ordinal_mod_stan_v2.stan");

mod <- cmdstan_model("./ordinal_mod_stan_v2.stan");
s_generate <- function(beta_1=0.5,beta_2=0.4, beta_3=0.2,#controls the contribution from the two sources of variation. 
                       sigma_beta_0=0.2, sigma_beta_1=0.1,  
                       basestudy= c(.31, .29, .20, .20),
                       n_train = 400) {
  
  def <- defData(varname = "A", formula = "1;1", dist = "trtAssign")
  def <- defData(def, varname="cov_1", formula = "0", 
                 variance= 1, dist="normal")  
  def <- defData(def, varname="beta_0_ind",formula = "0", 
                 variance= "..sigma_beta_0", dist="normal") 
  def <- defData(def, varname="beta_1_ind",formula = "0", 
                 variance= "..sigma_beta_1", dist="normal")  
  
  #define canonical parameter for 3 outcomes: 
  def <- defData(def, varname = "y_1_hat", 
                 formula = "..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind + ..beta_3*cov_1 + beta_1_ind*cov_1) ",
                 link="nonrandom")
  def <- defData(def,varname = "y_2_hat",   
                 formula = "..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind  + ..beta_3*cov_1 + beta_1_ind*cov_1)",
                 link="nonrandom")
  def <- defData(def,varname = "y_3_hat",   
                 formula = "..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind + ..beta_3*cov_1 + beta_1_ind*cov_1)",
                 link="nonrandom")
  #---Generate data---#
  ds <- genData(n_train, def)
  dd <- genOrdCat(ds,adjVar = "y_1_hat", basestudy, catVar = "y_1")
  dd <- genOrdCat(dd,adjVar = "y_2_hat", basestudy, catVar = "y_2")
  dd <- genOrdCat(dd,adjVar = "y_3_hat", basestudy, catVar = "y_3")
}

mvo_model <- function(iter, generated_data,mod){
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
  x <- model.matrix( ~ cov_1, data = generated_data)[,-1]
  y <- generated_data[,c("y_1","y_2","y_3")]
  
  D <- 3
  L <- length(unique(generated_data$y_1))
  studydata <- list(
    N = nrow(generated_data), y=y,
    x=x,A=generated_data$A, D=D, L=L)
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
    adapt_delta = 0.9)
  
  diagnostics_df <- as_draws_df(fit$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##Get posterior draws of all parameters
  draws_dt <- data.frame(as_draws_df(fit$draws()))
  
 
  #fixed effect in the interaction term for the first outcome
  beta_1 <- draws_dt$beta_1 
  beta_2 <- draws_dt$beta_2
  beta_3 <- draws_dt$beta_3
  sigma_beta_0 <- draws_dt$sigma_beta_0
  sigma_beta_1 <- draws_dt$sigma_beta_1
  tau_1 <- draws_dt$tau.1.
  tau_2 <- draws_dt$tau.2.
  tau_3 <- draws_dt$tau.3.
  
  res <- data.table(beta_1,beta_2,beta_3,
                    sigma_beta_0,sigma_beta_1,tau_1,tau_2,tau_3,
                    div,tree_hit)
  res_m <- round(apply(res,2,mean),3)
  
  return(data.table(iter=iter,beta_1=res_m[1],
                    beta_2=res_m[2],beta_3=res_m[3],
                    sigma_beta_0=res_m[4],sigma_beta_1=res_m[5],
                    tau_1=res_m[6],tau_2=res_m[7],tau_3=res_m[8],
                    div=res_m[9],tree_hit=res_m[10]
  ))
}
s_train <- function(iter, beta_1=0.5,beta_2=0.4, beta_3=0.2,#controls the contribution from the two sources of variation. 
                    sigma_beta_0=0.2, sigma_beta_1=0.1,
                    n_train = 400,mod){
  generated_data <- s_generate(beta_1=beta_1,beta_2=beta_2,beta_3=beta_3,#controls the contribution from the two sources of variation. 
                               sigma_beta_0=sigma_beta_0, sigma_beta_1=sigma_beta_1,
                               n_train = n_train)
  
  #mvo model
  mvo_results <- mvo_model(iter=iter,generated_data,mod)
  return(mvo_results)
}
# bayes_result <- rbindlist(lapply(1, function(x) s_train(x,beta_1=0.5,beta_2=0.4, beta_3=0.2,#controls the contribution from the two sources of variation.
#                                                            sigma_beta_0=0.2, sigma_beta_1=0.1,
#                                                            n_train = 400,mod=mod)))
# 
# save(bayes_result,file="./bayes_result.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

job <- Slurm_lapply(
  X = 1:100,
  FUN = s_train,
  beta_1=0.5,
  beta_2=0.4, 
  beta_3=0.2,#controls the contribution from the two sources of variation. 
  sigma_beta_0=0.2, 
  sigma_beta_1=0.1,
  n_train = 400,mod=mod,
  njobs = 40,
  mc.cores = 4L,
  job_name = "mvo_7",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "10:00:00", partition = "cpu_short", `mem-per-cpu` = "5G"),
  export = c("s_generate", "mvo_model"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/ordinal_model_res_v3.rda")




####--plot---#####
# bayes_result <- res
# gener_y1 <- data.frame(iter=1,
#                        beta_0= -0.7,
#                        beta_1=0.5,
#                        beta_2=0.4,
#                        beta_3=0.2,#controls the contribution from the two sources of variation.
#                        sigma_beta_0=0.2,
#                        sigma_beta_1=0.1,
#                        tau_1= -0.8,
#                        tau_2= 0.41,
#                        tau_3= 1.39,
#                        Md="True value")
# bayes_result <- bayes_result[bayes_result[,"tree_hit"] < 100,]
# M_MVO <- subset(bayes_result,select=-c(div, tree_hit))
# M_MVO$Md <- "Estimated"
# D_all <-rbind(M_MVO,gener_y1)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="MVO: Posterior mean of paramters",
#        y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
# 
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
