#=
This script generates the two-point correlator of the Haldane-Shastry model. 

Author: Lucas Z. Brito
=#

using JLD
using Plots 
using DelimitedFiles
include("./src/hs_tools.jl")

n = 21
wf_cm = load("./data/hs_data/cm_wf_n=$n.jld")
hs_wf = wf_cm["wf"]

l = 8
corr_ops = [chain_avg(x -> corr([ssz, ssz], [x, wrap_index(n, x + l)], n), n) for l in 1:l]

corrs = [expect(corr_op, hs_wf) for corr_op in corr_ops]
corrs_real = real.(corrs)
plot(corrs_real)

writedlm("../gutzwiller-recon/data/gutz_sz-sz_n=$n.txt", corrs_real)