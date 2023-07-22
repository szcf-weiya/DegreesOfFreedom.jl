# Splines

!!! note
    For simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. 


```@example
using DegreesOfFreedom
df_splines(N = 2, nrep = 10)
```

where `nrep` is the number of Monte Carlo samples to estimate the degrees of freedom and `N` is the number of repetitions. The columns are `estimated df`, `standard deviation of df` and `theoretical df`, respectively.