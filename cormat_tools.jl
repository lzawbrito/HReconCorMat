include("su2_tools.jl")

# makes an initial MPS for spin s and size Ns
function eval_SU2_Cormat(s::Float64, Ns::Integer, J_arr::Vector{Float64})
    nSpin = Int64(2*s + 1)
    H_onesite, H_op_vec = H_SU2(s, J_arr) # make the onsite term

    H_mpo = makeMPO(H_onesite,nSpin,Ns); # make the MPO!
    psi0 = makePsi0(s,Ns);
    psi = copy(psi0)

    # Make array of operators
    nOps = length(H_op_vec)
    MPO_op_vec = []
    for idx = 1:nOps
        MPO_here = makeMPO(H_op_vec[idx],nSpin,Ns);
        push!(MPO_op_vec, MPO_here)
    end

    # do DMRG
    psi, variance = dmrg_anneal(psi, H_mpo)

    # evaluate Correlation matrix
    corMat = calc_cor_mat(psi, MPO_op_vec)

    return corMat, psi, variance
    
end

function estimate_H(corMat, drop_idx)
    
    nOps = size(corMat)[1]
    keep_idx = setdiff(1:nOps, drop_idx) # operators to keep in evaluating corMat eigenpairs
    corMat_reduced = corMat[keep_idx,keep_idx]
    
    vals,vecs = LinearAlgebra.eigen(corMat_reduced)
    min_eig, tar_idx = findmin(broadcast(abs,vals))
    H_est = broadcast(real,vecs[:,tar_idx])/maximum(broadcast(abs,vecs[:,tar_idx]))

    min_eig, H_est, vals, vecs

end
    

function calc_cor_mat(psi, MPO_op_vec)
    # evaluate Correlation matrix
    nOps = length(MPO_op_vec)
    corMat = zeros(ComplexF64,nOps,nOps)
    op_expect = zeros(ComplexF64,nOps)

    for idx = 1:nOps
        op_expect[idx] = expect(psi,MPO_op_vec[idx])
    end

    for idx1 = 1:nOps
        for idx2 = 1:nOps
            val_h = 0.5*expect(psi,MPO_op_vec[idx1],MPO_op_vec[idx2])
            val_h += 0.5*expect(psi,MPO_op_vec[idx2],MPO_op_vec[idx1])
            val_h += -(op_expect[idx1]*op_expect[idx2])
            corMat[idx1,idx2] = val_h
        end
    end

    return corMat

end

function dmrg_anneal(psi, H_mpo)
    # DMRG sweeps, with refinement 
    nsweeps_m10 = 40
    nsweeps_m40 = 20
    nsweeps_m80 = 10

    for i in 1:nsweeps_m10
        dmrg(psi, H_mpo, maxm = 10, cutoff = 1E-4)
    end

    for i in 1:nsweeps_m40
        dmrg(psi, H_mpo, maxm = 40, cutoff = 1E-8)
    end

    for i in 1:nsweeps_m80
        dmrg(psi, H_mpo, maxm = 80, cutoff = 1E-15)
    end
    
    variance = expect(psi,H_mpo,H_mpo)- (expect(psi, H_mpo))^2
    display(real(variance))

    return psi, variance

end
