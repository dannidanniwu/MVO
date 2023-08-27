# A Bayesian Multivariate Hierarchical Model for Developing a Treatment Benefit Index

## Background
Precision medicine has led to the development of targeted treatment strategies tailored to individual patients based on their characteristics and disease manifestations. Although precision medicine often focuses on a single health outcome for individualized treatment decision rules (ITRs), relying only on a single outcome rather than all available outcomes information leads to suboptimal data usage when developing optimal ITRs.

## Methods
To address this limitation, we propose a Bayesian multivariate hierarchical model that leverages the wealth of correlated health outcomes collected in clinical trials. The approach jointly models mixed types of correlated outcomes, facilitating the ``borrowing of information" across the multivariate outcomes, and results in a more accurate estimation of heterogeneous treatment effects compared to using single regression models for each outcome. We develop a treatment benefit index, which quantifies the relative treatment benefit of the experimental treatment over the control treatment, based on the proposed multivariate outcome model.

##  Results
We demonstrate the strengths of the proposed approach through extensive simulations and an application to an international Coronavirus Disease 2019 (COVID-19) treatment trial. Simulation results indicate that the proposed method reduces the occurrence of erroneous treatment decisions compared to a single regression model for a single health outcome. Additionally, the sensitivity analysis demonstrates the robustness of the model across various study scenarios. Application of the method to the COVID-19 trial exhibits improvements in estimating the individual-level treatment efficacy (indicated by narrower credible intervals for odds ratios) and optimal ITRs.

## Conclusion
The study jointly models mixed types of outcomes in the context of developing ITRs. By considering multiple health outcomes, the proposed approach can advance the development of more effective and reliable personalized treatment recommendations. 
