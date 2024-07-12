# Splines

!!! note
    For simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. 


```@example
using DegreesOfFreedom
df_splines(n = 20, nrep = 2, nMC = 10)
```

where 

- `n` is the sample size
- `nMC` is the number of Monte Carlo samples to estimate the degrees of freedom
- `nrep` is the number of repetitions. 

These three columns are `estimated df`, `standard deviation of df` and `theoretical df`, respectively.