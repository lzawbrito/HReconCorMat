# Author: Stephen Carr

# USING DMRJulia v0.8.7
# (most recent version tries to call BLAS stev! for complex psi... doesn't work!)
# NEED TO EDIT (to make complex work)
# lib/contractions.jl:
#   change line 305 from:
#       mA,mB = checkType(A,B)
#   to
#       mA = A
#       mB = B
#

#using MKL # comment out for non-intel machines, will use openBLAS by default
using Printf, FileIO
include("cormat_tools.jl")

# setup system with isotropic J
spinmag = 1.0;
nSpins = Int(spinmag*2 + 1)
Ns = 20 # number of sites
nruns = 10 # number of runs (random samples of J_arr)
fname = "dat_spin1_sweep_run.jld2"

nSpin = Int64(spinmag*2 + 1) # dimension of spin matrices
max_p = nSpin-1 # maximum unique power of (S_i*S_j)^p terms


dataset = Vector{Any}(undef,nruns)

for run_idx = 1:nruns
    #J_arr = 2.0*(rand(max_p,1) .- 0.5) # random values between [-1,1]
    #J_arr = rand(max_p) # random values between [0,1]
    #J_arr = J_arr ./ maximum(broadcast(abs,J_arr)) # normalize to biggest value

    #min_J = findmin(J_arr)[1]
    #drop_idx = findmin(J_arr)[2]

    # generate J_arr so that two value are > 0.7, and one is < 0.2
    max_idx = rand(1:nSpins,1)[1]
    min_idx = rand( setdiff(1:nSpins,max_idx),1)[1]
    J_arr = zeros(3) .+ 0.7 .+ 0.2*rand(1) # Middle value for all other terms
    J_arr[max_idx] = 1.0 # max value
    J_arr[min_idx] = 0.2*rand(1)[1] # minimum value
    min_J = J_arr[min_idx]
    drop_idx = min_idx

    rundata = Dict("J_arr" => J_arr, "min_J" => min_J, "drop_idx" => drop_idx)

    # calculate correlation matrix
    @time corMat, psi, variance = eval_SU2_Cormat(spinmag, Ns, J_arr)
    #println("post-DMRG psi variance: ",variance)

    rundata["corMat"] = corMat
    rundata["variance"] = variance

    # find eigenvalues of corMat
    drop_idxs = []
    min_eig, H_est, vals, vecs = estimate_H(corMat, drop_idxs)

    rundata["eigs_full"] = vals
    rundata["vecs_full"] = vecs
    rundata["min_eig_full"] = min_eig
    rundata["H_est_full"] = H_est

    drop_idxs = [drop_idx]
    min_eig, H_est, vals, vecs = estimate_H(corMat, drop_idxs)

    rundata["eigs_reduc"] = vals
    rundata["vecs_reduc"] = vecs
    rundata["min_eig_reduc"] = min_eig
    rundata["H_est_reduc"] = H_est

    dataset[run_idx] = rundata

end

save(fname, "dataset", dataset)

#=
tar_run = dataset[1]

display("J_arr:")
display(tar_run["J_arr"])

@printf("eigenvalues = ")
for val in tar_run["eigs_full"]
    @printf("%.5f ",real(val))
end
@printf("\n")

@printf("min_eig = %E \n",tar_run["min_eig_full"])

@printf("operators = ")
for op in tar_run["H_est_full"]
    @printf("%.5f ",op)
end
@printf("\n")

@printf("dropping index: %d \n", tar_run["drop_idx"])
@printf("eigenvalues [reduc] = ")
for val in tar_run["eigs_reduc"]
    @printf("%.5f ",real(val))
end
@printf("\n")

@printf("min_eig [reduc] = %E \n",tar_run["min_eig_reduc"])

@printf("operators [reduc]= ")
for op in tar_run["H_est_reduc"]
    @printf("%.5f ",op)
end
@printf("\n")
=#



