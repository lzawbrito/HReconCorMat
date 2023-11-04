# Author: Stephen Carr

using FileIO
using Plots

dataset = load("SU2_sweep_run.jld2","dataset")

nruns = length(dataset)
min_J = zeros(nruns)
min_eig_full = zeros(nruns)
min_eig_reduc = zeros(nruns)
spectral_est = zeros(nruns)

for idx = 1:nruns
    dat = dataset[idx]
    min_J[idx] = dat["min_J"]
    min_eig_full[idx] = dat["min_eig_full"]
    min_eig_reduc[idx] = dat["min_eig_reduc"]
    corMat = broadcast(real,dat["corMat"])
    display(dat["J_arr"])
    display(corMat)
    drop_idx = dat["drop_idx"]
    keep_idx = setdiff(1:3,drop_idx)

    diag_drop_val = diag(corMat)[drop_idx]
    diag_kept_val = maximum(broadcast(abs,diag(corMat)[keep_idx]))
    offdiag_drop_val = sum(broadcast(abs,corMat[drop_idx,keep_idx]))
    spectral_est[idx] = 0.01*abs(offdiag_drop_val^2/(diag_drop_val))
end

scatter(min_J,spectral_est)
xlabel!("min J")
ylabel!("spectral estimate")