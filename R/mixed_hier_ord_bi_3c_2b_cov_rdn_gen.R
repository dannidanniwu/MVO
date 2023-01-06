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
mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/mixed_hier_ord_binary_2cov_v3.stan");


#mod <- cmdstan_model("./mixed_hier_ord_binary_2cov_v3.stan");
s_generate <- function(sigma_beta_trt=0.2, sigma_beta_1=0.1,
                       sigma_beta_2 = 0.1,sigma_beta_3 = 0.09,
                       sigma_beta_4 = 0.12,sigma_beta_5 = 0.09,
                       basestudy= c(.31, .29, .20, .20),
                       n_train = 400) {
  
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
  #main effect of covariates
  m_cov1 <- rnorm(4,mean=0.4,sd=0.02)
  m_cov1_g1 <- m_cov1[1]
  m_cov1_g2 <- m_cov1[2]
  m_cov1_g3 <- m_cov1[3]
  m_cov1_g4 <- m_cov1[4]
  
  m_cov2 <- rnorm(4,mean=-0.4,sd=0.02)
  m_cov2_g1 <- m_cov2[1]
  m_cov2_g2 <- m_cov2[2]
  m_cov2_g3 <- m_cov2[3]
  m_cov2_g4 <- m_cov2[4]
  
  m_cov3 <- rnorm(4,mean=0.15,sd=0.02)
  m_cov3_g1 <- m_cov3[1]
  m_cov3_g2 <- m_cov3[2]
  m_cov3_g3 <- m_cov3[3]
  m_cov3_g4 <- m_cov3[4]
  
  m_cov4 <- rnorm(4,mean=0.2,sd=0.02)
  m_cov4_g1 <- m_cov4[1]
  m_cov4_g2 <- m_cov4[2]
  m_cov4_g3 <- m_cov4[3]
  m_cov4_g4 <- m_cov4[4]
  
  m_cov5 <- rnorm(4,mean=-0.2,sd=0.02)
  m_cov5_g1 <- m_cov5[1]
  m_cov5_g2 <- m_cov5[2]
  m_cov5_g3 <- m_cov5[3]
  m_cov5_g4 <- m_cov5[4]
  
  #main effect of treatment
  beta_trt <- rnorm(4,mean=0.38,sd=0.02)
  beta_trt_g1 <- beta_trt[1]
  beta_trt_g2 <- beta_trt[2]
  beta_trt_g3 <- beta_trt[3]
  beta_trt_g4 <- beta_trt[4]
  
  #interaction effect
  beta_int1 <- rnorm(4,mean=0.15,sd=0.01)
  beta_int1_g1 <- beta_int1[1]
  beta_int1_g2 <- beta_int1[2]
  beta_int1_g3 <- beta_int1[3]
  beta_int1_g4 <- beta_int1[4]
  
  beta_int2 <- rnorm(4,mean=-0.09,sd=0.01)
  beta_int2_g1 <- beta_int2[1]
  beta_int2_g2 <- beta_int2[2]
  beta_int2_g3 <- beta_int2[3]
  beta_int2_g4 <- beta_int2[4]
  
  beta_int3 <- rnorm(4,mean=0.09,sd=0.01)
  beta_int3_g1 <- beta_int3[1]
  beta_int3_g2 <- beta_int3[2]
  beta_int3_g3 <- beta_int3[3]
  beta_int3_g4 <- beta_int3[4]
  
  beta_int4 <- rnorm(4,mean=0.05,sd=0.01)
  beta_int4_g1 <- beta_int4[1]
  beta_int4_g2 <- beta_int4[2]
  beta_int4_g3 <- beta_int4[3]
  beta_int4_g4 <- beta_int4[4]
  
  beta_int5 <- rnorm(4,mean=-0.04,sd=0.01)
  beta_int5_g1 <- beta_int5[1]
  beta_int5_g2 <- beta_int5[2]
  beta_int5_g3 <- beta_int5[3]
  beta_int5_g4 <- beta_int5[4]
  
  beta_0 <- rnorm(3,mean=-0.3,sd=0.01)
  beta_0_g2 <- beta_0[1]
  beta_0_g3 <- beta_0[2]
  beta_0_g4 <- beta_0[3]
  
  def <- defData(def, varname="beta_trt_ind",formula = 0, 
                 variance= "..sigma_beta_trt", dist="normal") 
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
                 formula = "..m_cov1_g1 *cov_1 + ..m_cov2_g1*cov_2 + ..m_cov3_g1*cov_3 + ..m_cov4_g1*cov_4 +..m_cov5_g1*cov_5 +
                      A*(..beta_trt_g1 + beta_trt_ind + ..beta_int1_g1*cov_1 + ..beta_int2_g1*cov_2 +  ..beta_int3_g1*cov_3 + 
                      + ..beta_int4_g1* cov_4 + ..beta_int5_g1*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3+ beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="nonrandom")
  def <- defData(def, varname = "y_2",dist="binary",  
                 formula = "..beta_0_g2 + ..m_cov1_g2 *cov_1 + ..m_cov2_g2*cov_2 + ..m_cov3_g2*cov_3 + ..m_cov4_g2*cov_4 +..m_cov5_g2*cov_5 +
                      A*(..beta_trt_g2 + beta_trt_ind + ..beta_int1_g2*cov_1 + ..beta_int2_g2*cov_2 +  ..beta_int3_g2*cov_3 + 
                      + ..beta_int4_g2* cov_4 + ..beta_int5_g2*cov_5+ 
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  
  def <- defData(def, varname = "y_3",dist="binary",  
                 formula = "..beta_0_g3 + ..m_cov1_g3 *cov_1 + ..m_cov2_g3*cov_2 + ..m_cov3_g3*cov_3 + ..m_cov4_g3*cov_4 +..m_cov5_g3*cov_5
                      A*(..beta_trt_g3 + beta_trt_ind + ..beta_int1_g3*cov_1 + ..beta_int2_g3*cov_2 +  ..beta_int3_g3*cov_3 + 
                      + ..beta_int4_g3* cov_4 + ..beta_int5_g3*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  def <- defData(def, varname = "y_4",dist="binary",  
                 formula = "..beta_0_g4 + ..m_cov1_g4 *cov_1 + ..m_cov2_g4*cov_2 + ..m_cov3_g4*cov_3 + ..m_cov4_g4*cov_4 +..m_cov5_g4*cov_5 +
                      A*(..beta_trt_g4 + beta_trt_ind + ..beta_int1_g4*cov_1 + ..beta_int2_g4*cov_2 +  ..beta_int3_g4*cov_3 + 
                      + ..beta_int4_g4* cov_4 + ..beta_int5_g4*cov_5+
                 beta_1_ind*cov_1 + beta_2_ind*cov_2 + beta_3_ind*cov_3 + beta_4_ind*cov_4 + beta_5_ind*cov_5)",
                 link="logit")
  #---Generate data---#
  ds <- genData(n_train, def)
  dd <- genOrdCat(ds,adjVar = "y_1_hat", basestudy, catVar = "y_ord")
  dd[,`:=`(m_cov1_g1=m_cov1_g1, m_cov2_g1 = m_cov2_g1, m_cov3_g1 = m_cov3_g1,
           m_cov4_g1 = m_cov4_g1,m_cov5_g1 = m_cov5_g1,
           beta_trt_g1=beta_trt_g1, beta_int1_g1=beta_int1_g1,beta_int2_g1=beta_int2_g1,
           beta_int3_g1=beta_int3_g1,beta_int4_g1=beta_int4_g1,
           beta_int5_g1=beta_int5_g1
           )]
}

mvo_model <- function(iter, generated_data,mod){
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
  
  
  #fixed effect in the interaction term for the first outcome
  beta_0 <- draws_dt$beta_0
  beta_cov1 <- draws_dt$beta_1.1.4#covariate main effect
  beta_cov2 <- draws_dt$beta_1.2.4
  beta_cov3 <- draws_dt$beta_1.3.4
  beta_cov4 <- draws_dt$beta_1.4.4
  beta_cov5 <- draws_dt$beta_1.5.4
  
  
  beta_trt <- draws_dt$beta.1.4.
  beta_inter_cov1 <- draws_dt$beta.2.4.
  beta_inter_cov2 <- draws_dt$beta.3.4.
  beta_inter_cov3 <- draws_dt$beta.4.4.
  beta_inter_cov4 <- draws_dt$beta.5.4.
  beta_inter_cov5 <- draws_dt$beta.6.4.
  
  tau_1 <- draws_dt$tau.1.
  tau_2 <- draws_dt$tau.2.
  tau_3 <- draws_dt$tau.3.
  
  sigma_beta_trt <- draws_dt$sigma_beta.1.
  sigma_beta_cov1 <- draws_dt$sigma_beta.2.
  sigma_beta_cov2 <- draws_dt$sigma_beta.3.
  sigma_beta_cov3 <- draws_dt$sigma_beta.4.
  sigma_beta_cov4 <- draws_dt$sigma_beta.5.
  sigma_beta_cov5 <- draws_dt$sigma_beta.6.
  # Phi_1 <- draws_dt[,c(grep("^Phi_c.1",colnames(draws_dt),value=TRUE))]
  # Phi_2 <- draws_dt[,c(grep("^Phi_c.2",colnames(draws_dt),value=TRUE))]
  # Phi_3 <- draws_dt[,c(grep("^Phi_c.3",colnames(draws_dt),value=TRUE))]
  #overall mean of betas: beta^*
  beta_star_trt <- draws_dt[,c(grep("^beta_star.1",colnames(draws_dt),value=TRUE))]
  beta_star_cov1 <- draws_dt[,c(grep("^beta_star.2",colnames(draws_dt),value=TRUE))]
  beta_star_cov2 <- draws_dt[,c(grep("^beta_star.3",colnames(draws_dt),value=TRUE))]
  beta_star_cov3 <- draws_dt[,c(grep("^beta_star.4",colnames(draws_dt),value=TRUE))]
  beta_star_cov4 <- draws_dt[,c(grep("^beta_star.5",colnames(draws_dt),value=TRUE))]
  beta_star_cov5 <- draws_dt[,c(grep("^beta_star.6",colnames(draws_dt),value=TRUE))]
  
  res <- data.table(beta_0,beta_cov1,beta_cov2,beta_cov3,beta_cov4,beta_cov5,
                    beta_trt,beta_inter_cov1,
                    beta_inter_cov2,beta_inter_cov3,beta_inter_cov4,beta_inter_cov5,
                    tau_1,tau_2,tau_3,beta_star_trt,beta_star_cov1,
                    beta_star_cov2,beta_star_cov3, 
                    beta_star_cov4,beta_star_cov5, 
                    div,tree_hit,sigma_beta_trt,
                    sigma_beta_cov1,sigma_beta_cov2,sigma_beta_cov3,sigma_beta_cov4,
                    sigma_beta_cov5)
  res_m <- round(apply(res,2,mean),3)
  
  #check proportion of patients belong to each level of outcomes
  p_1_g1 <- mean(generated_data$y_ord %in% 1)
  p_2_g1 <- mean(generated_data$y_ord %in% 2)
  p_3_g1 <- mean(generated_data$y_ord %in% 3)
  p_4_g1 <- mean(generated_data$y_ord %in% 4)
  
  p_1_g2 <- mean(generated_data$y_2 %in% 1)
  p_1_g3 <- mean(generated_data$y_3 %in% 1)
  p_1_g4 <- mean(generated_data$y_4 %in% 1)
  
  
  return(data.table(cbind(iter,t(res_m),
                          m_cov1_g1_true = unique(generated_data$m_cov1_g1),
                          m_cov2_g1_true = unique(generated_data$m_cov2_g1),
                          m_cov3_g1_true = unique(generated_data$m_cov3_g1),
                          m_cov4_g1_true = unique(generated_data$m_cov4_g1),
                          m_cov5_g1_true = unique(generated_data$m_cov5_g1),
                          beta_trt_g1_true = unique(generated_data$beta_trt_g1),
                          beta_int1_g1_true = unique(generated_data$beta_int1_g1),
                          beta_int2_g1_true = unique(generated_data$beta_int2_g1),
                          beta_int3_g1_true = unique(generated_data$beta_int3_g1),
                          beta_int4_g1_true = unique(generated_data$beta_int4_g1),
                          beta_int5_g1_true = unique(generated_data$beta_int5_g1),
                          p_1_g1=p_1_g1,
                          p_2_g1=p_2_g1,
                          p_3_g1=p_3_g1,
                          p_4_g1=p_4_g1,
                          p_1_g2= p_1_g2,
                          p_1_g3= p_1_g3,
                          p_1_g4= p_1_g4
                          )))
}
s_train <- function(iter, sigma_beta_trt=0.2, sigma_beta_1=0.1,
                    sigma_beta_2 = 0.1,sigma_beta_3 = 0.09,
                    sigma_beta_4 = 0.12,sigma_beta_5 = 0.09,
                    basestudy= c(.31, .29, .20, .20),
                    n_train = 400, mod=mod){
  generated_data <- s_generate(basestudy= basestudy,sigma_beta_trt=sigma_beta_trt, 
                               sigma_beta_1=sigma_beta_1, sigma_beta_2=sigma_beta_2, 
                               sigma_beta_3 = sigma_beta_3, sigma_beta_4=sigma_beta_4, 
                               sigma_beta_5 = sigma_beta_5, 
                               n_train = n_train)
  
  #mvo model
  mvo_results <- mvo_model(iter=iter,generated_data,mod)
  return(mvo_results)
}
# bayes_result <- rbindlist(lapply(1, function(x) s_train(x,sigma_beta_trt=0.2, sigma_beta_1=0.1,
#                                                         sigma_beta_2 = 0.1,sigma_beta_3 = 0.09,
#                                                         sigma_beta_4 = 0.12,sigma_beta_5 = 0.09,
#                                                         basestudy= c(.31, .29, .20, .20),
#                                                         n_train = 400,mod=mod)))

# save(bayes_result,file="./bayes_result.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

job <- Slurm_lapply(
  X = 1:100,
  FUN = s_train,
  sigma_beta_trt=0.2, sigma_beta_1=0.1,
  sigma_beta_2 = 0.1,sigma_beta_3 = 0.09,
  sigma_beta_4 = 0.12,sigma_beta_5 = 0.09,
  basestudy= c(.31, .29, .20, .20),
  n_train = 400, mod=mod,
  njobs = 50,
  mc.cores = 4L,
  job_name = "mvo_37",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "8:00:00", partition = "cpu_short", `mem-per-cpu` = "5G"),
  export = c("s_generate", "mvo_model"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/mixed_hier_ord_bi_3c_2b_cov_rdn_gen_v3.rda")

#for n=900
bayes_result <- rbind(res,res_2)
bayes_result$iter <- 1:dim(bayes_result)[1]

####--plot---#####
bayes_result <- res[res$div <=100,]
dim(bayes_result)#100
#the ordinal outcome


gener_y4 <- data.frame(iter=1,
                       beta_0=-0.3,
                      beta_cov1= bayes_result$m_cov1_g1_true,
                       beta_cov2 = bayes_result$m_cov2_g1_true,
                       beta_cov3 = bayes_result$m_cov3_g1_true,
                       beta_cov4 = bayes_result$m_cov4_g1_true,
                       beta_cov5 = bayes_result$m_cov5_g1_true,
                       beta_trt = bayes_result$beta_trt_g1_true,
                       beta_inter_cov1=bayes_result$beta_int1_g1_true,
                       beta_inter_cov2=bayes_result$beta_int2_g1_true,
                       beta_inter_cov3=bayes_result$beta_int3_g1_true,
                       beta_inter_cov4=bayes_result$beta_int4_g1_true,
                       beta_inter_cov5=bayes_result$beta_int5_g1_true,
                       beta_star_trt=0.38,
                       beta_star_cov1=0.15,
                       beta_star_cov2=-0.09,
                       beta_star_cov3=0.09,
                       beta_star_cov4= 0.05,
                       beta_star_cov5= -0.04,
                       tau_1= -0.8,
                       tau_2= 0.41,
                       tau_3= 1.39,
                      sigma_beta_trt=0.02,
                       sigma_beta_cov1=0.01, sigma_beta_cov2=0.01,
                       sigma_beta_cov3 = 0.01,sigma_beta_cov4 = 0.01,
                       sigma_beta_cov5 = 0.01,
                       Md="True value")
M_MVO <- subset(bayes_result,select=c(iter,beta_0,beta_cov1,beta_cov2,beta_cov3,
                                      beta_cov4,beta_cov5,
                                      beta_trt,beta_inter_cov1,
                                      beta_inter_cov2,beta_inter_cov3,
                                      beta_inter_cov4,beta_inter_cov5,
                                      tau_1,tau_2,tau_3,beta_star_trt,beta_star_cov1,
                                      beta_star_cov2,
                                      beta_star_cov3,beta_star_cov4,beta_star_cov5,
                                      sigma_beta_trt,
                                      sigma_beta_cov1,sigma_beta_cov2,sigma_beta_cov3,sigma_beta_cov4,
                                      sigma_beta_cov5))
M_MVO$Md <- "Estimated"
D_all <-rbind(M_MVO,gener_y4)

M_data <- reshape2::melt(D_all,id=c("iter","Md"))

ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
  geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
  labs(title="MVO: Posterior mean of paramters (n=400, prior of sigma_j ~ exp(mean=0.01))",
       y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
  theme(strip.text.x = element_blank())+labs(fill="CLASS")

# # ###---Check data generation---#
# odds <- function(x){
#   y = 1/(1-x)
#   return(y)
# }
# logOdds.upexp <- log(odds(cumsum(dd[A==0,prop.table(table(y_1))])))
# 
# library(ordinal)
# clmFit <- clmm2(y_1 ~ cov_1 + A + A*cov_1,data=dd)
# summary(clmFit)
