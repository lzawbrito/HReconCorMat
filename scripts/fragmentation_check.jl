#=
This script verifies the claim that there exists operators with zero variance 
wrt to an eigenstate of H that nonetheless do not commute with H. This is done 
in context of the t-J_z model (see Moudalya and Motrunich 2022).

Author: Lucas Z. Brito
=#

using LinearAlgebra
using Combinatorics: combinations
using SparseArrays
using JLD
using Arpack

include("./src/hs_tools.jl")

# Creation operators
cp_u = begin m = zeros(ComplexF64, 4, 4); m[2, 1] = 1; m[4, 3] = 1; m end |> sparse
cp_d = begin m = zeros(ComplexF64, 4, 4); m[3, 1] = 1; m[4, 2] = 1; m end |> sparse
	
# Annihilation operators
cm_u = sparse(cp_u')
cm_d = sparse(cp_d')

# Up and down spin number operators
n_u = cp_u * cm_u 
n_d = cp_d * cm_d 

# Total number operator
n_tot = n_u + n_d

cm_tilde_u = cm_u * (sparse(I, 4, 4) - n_d)
cm_tilde_d = cm_d * (sparse(I, 4, 4) - n_u)
cp_tilde_u = sparse(cm_tilde_u')
cp_tilde_d = sparse(cm_tilde_d')

sz_c = cp_tilde_u * cm_tilde_u - cp_tilde_d * cm_tilde_d

n_tilde_u = cp_tilde_u * cm_tilde_u
n_tilde_d = cp_tilde_d * cm_tilde_d

n_tot_tilde_u(n) = sum(i -> op_in_subspace(cp_tilde_u * cm_tilde_u, i, n), 1:n)
n_tot_tilde_d(n) = sum(i -> op_in_subspace(cp_tilde_d * cm_tilde_d, i, n), 1:n)

t_coupling(i, j) = abs(i - j)^(-2)

function tjz_model(n::Int)
	h = sparse(zeros(4^n, 4^n))

	for (i, j) in combinations(1:n, 2)
		h += - op_in_subspace(cm_tilde_u, i, n) * op_in_subspace(cp_tilde_u, j, n)
		h += - op_in_subspace(cm_tilde_u, j, n) * op_in_subspace(cp_tilde_u, i, n)
		h += - op_in_subspace(cm_tilde_d, i, n) * op_in_subspace(cp_tilde_d, j, n)
		h += - op_in_subspace(cm_tilde_d, j, n) * op_in_subspace(cp_tilde_d, i, n)
		h = h * t_coupling(i, j)
		h += op_in_subspace(sz_c, i, n) * op_in_subspace(sz_c, j, n)
	end

	for i in 1:n 
		h += op_in_subspace(sz_c, i, n) + op_in_subspace(sz_c, i, n)^2
	end

	return h
end

################################################################################

n = 5

h = tjz_model(n)
h_evals, h_evecs = eigen(Matrix(h))
tjz_wf = normalize(sparse(h_evecs[:, 1]))
maximum(real.(abs.(tjz_wf)))

# The below is an attempt to construct the correlation matrix. 
# op_basis = [begin 
# 					op_in_subspace(cm_tilde_u, 1, n) * op_in_subspace(cp_tilde_u, i, n) 
# 					+ op_in_subspace(cm_tilde_u, i, n) * op_in_subspace(cp_tilde_u, 1, n) 
# 					+ op_in_subspace(cm_tilde_d, 1, n) * op_in_subspace(cp_tilde_d, i, n) 
# 					+ op_in_subspace(cm_tilde_d, i, n) * op_in_subspace(cp_tilde_d, 1, n)
# 				end for i in 2:n]


# append!(op_basis, [op_in_subspace(sz_c, 1, n) * op_in_subspace(sz_c, i, n) for i in 2:n])
# append!(op_basis, [op_in_subspace(sz_c, 1, n), op_in_subspace(sz_c, 1, n)^2])

# function make_tjz_cormat(op_basis, wf)
# 	indices = eachindex(op_basis)
# 	return [1/2 * expect(op_basis[i] * op_basis[j] + op_basis[j] * op_basis[i], wf) - expect(op_basis[i], wf) * expect(op_basis[j], wf) for i in indices, j in indices]
# end

# cm = make_tjz_cormat(op_basis, tjz_wf)
# cm = real(cm)
# evals, evecs = eigen(cm)
# evecs[:, 5]

# evecs[1, 5] * abs.(1:4).^(-2)

comm = op_in_subspace(n_tilde_u, 1, n) * h - h * op_in_subspace(n_tilde_u, 1, n)
maximum(abs.(comm))

# Commutant algebra integral of motion of eq (24) in Moudgalya and Motrunich (2022)
iom = let h = sparse(zeros(4^n, 4^n))
	for (i, j) in combinations(1:n, 2) 
		h += op_in_subspace(n_tilde_u, i, n) * op_in_subspace(n_tilde_d, j, n)
	end
	h
end	

# Check variance of iom vanishes 
variance_iom = expect(iom^2, tjz_wf) - expect(iom, tjz_wf)^2

# Check iom does not commute with h
maximum(abs.(iom * h - h * iom))