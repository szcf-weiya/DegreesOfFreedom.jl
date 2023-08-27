# DegreesOfFreedom.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://szcf-weiya.github.io/DegreesOfFreedom.jl/dev) [![codecov](https://codecov.io/gh/szcf-weiya/DegreesOfFreedom.jl/branch/master/graph/badge.svg?token=d3tsdGzbcy)](https://codecov.io/gh/szcf-weiya/DegreesOfFreedom.jl) [![CI](https://github.com/szcf-weiya/DegreesOfFreedom.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/szcf-weiya/DegreesOfFreedom.jl/actions/workflows/ci.yml)

Julia package for "Lijun Wang, Hongyu Zhao and Xiaodan Fan (2023). Degrees of Freedom: Search Cost and Self-consistency. *Manuscript*"

## :hammer_and_wrench: Installation

*DegreesOfFreedom.jl* is available at the General Registry, so you can easily install the package in the Julia session after typing `]`,

```julia
julia> ]
(@v1.8) pkg> add DegreesOfFreedom
```

## :books: Documentation

The documentation <https://hohoweiya.xyz/DegreesOfFreedom.jl/stable/> elaborates on the degrees of freedom of various popular machine learning methods, including Lasso, Best Subset, Regression Tree, Splines and MARS. 

## :rocket: R package

We also wrap up the correcting procedure for MARS as a standalone R package [earth.dof.patch](https://github.com/szcf-weiya/earth.dof.patch). By its name, it can be viewed as a patch package on the `earth` package for MARS. You can quickly try our approach after loading `earth` and `earth.dof.patch`. A toy example can be found [here](https://hohoweiya.xyz/earth.dof.patch/articles/MSE-Comparisons.html).