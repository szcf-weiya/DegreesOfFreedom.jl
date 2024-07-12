# Best subset vs Lasso

Reproduce the comparisons of degrees of freedom between the best subset and Lasso.

```@example
using DegreesOfFreedom
run_experiment_lasso_vs_subset(folder = ".", ylim = 12.5)
```

```@raw html
<style>
embed {
    height: 800px !important;
}
</style>

<embed src="../df_lasso_subset.pdf" width="800px" height="2100px" />
```