using LinearAlgebra
using SpecialFunctions: sinint
using Combinatorics: combinations
using SparseArrays
using JLD

SparseOrFull{T, Ti} = Union{SparseMatrixCSC{T, Ti}, Matrix{T}}

sx = 1/2 * ComplexF64[0 1; 1 0]
sy = 1/2 * ComplexF64[0 -im; im 0]
sz = 1/2 * ComplexF64[1 0; 0 -1]
ssx, ssy, ssz = map(sparse, [sx, sy, sz])

s_minus = [0 0; 1 0]
s_plus = [0 1; 0 0]

up, down = [1; 0], [0; 1]

corr_func(r)::Float64 = sinint(π * r) * ((-1) ^ r) / (4 * π * r)
z(x::Int, n::Int)::ComplexF64 = exp(2im * π * x / n)

expect(op, state)::ComplexF64 = (state' * op * state)[1, 1]

"""
Wraps in index with respect to periodicity of chain. If `j` provided, computes 
wrapped (absolute value) distance between `i` and `j`.

# Examples
```
julia> for i in 1:(3 + 1) println(wrap_index(3, i)) end 
1
2
3
1

julia> for i in 1:(3 + 1) println(wrap_index(3, i, 2)) end 
1
0
1
1
```
"""
function wrap_index(n::Int, i::Int, j::Int=0)::Int
	wrap = Int(abs(i - j) - 1) % n + 1 
	if j != 0 
		if wrap > n / 2
			return abs(n - abs(wrap))
		end
	end
	return wrap
end

function op_basis_dim(n)
	return trunc(Int, n / 2)
end

function op_basis_dim_oc(n)
	return length(combinations(1:n, 2))
end

function psi(x, n::Int)::ComplexF64
	f1 = 1 
	for xi in x
		f1 *= z(xi, n)
	end 

	f2 = 1 
	for (i, j) in combinations(1:length(x), 2)
		f2 *= (z(x[i], n) - z(x[j], n)) ^ 2
	end
	return f1 * f2
end

function wf(m::Int, n::Int)::Vector{ComplexF64}
	# Base case
	state = zeros(ComplexF64, 2 ^ n)
	for x in combinations(1:n, m)
		coeff = psi(x, n)

		# Find index corresponding to applying spin lowering operator on sites 
		# in x
		idx = 1
		for j in x
			idx += 2 ^ (n - j)
		end 
		zs = zeros(ComplexF64, 2 ^ n)
		zs[idx] = 1

		state += coeff * zs
	end
	return state
end

function op_in_subspace(op::SparseOrFull{ComplexF64, Int64}, 
						i::Int, n::Int)::SparseOrFull{ComplexF64, Int64}
	new_op = I(size(op)[1])

	if i == 1
		new_op = op
	end
	for k in 2:n
		if k == i
			new_op = kron(new_op, op)
		else
			new_op = kron(new_op, I(size(op)[1]))
		end
	end
	return new_op
end

function corr(ops::Union{Vector{SparseMatrixCSC{ComplexF64, Int64}},Vector{Matrix{ComplexF64}}}, 
			  sites::Vector{Int}, n::Int)::SparseOrFull{ComplexF64, Int64}
	@assert length(ops) == length(sites)
	op = I(2 ^ n) 
	for (current_op, site) in zip(ops, sites)
		op *= op_in_subspace(current_op, site, n)	
	end 
	return op
end

function dot_op(i::Int, j::Int, n::Int)::SparseOrFull{ComplexF64, Int64}
	op =  corr([ssx, ssx], [i, j], n)
	op += corr([ssy, ssy], [i, j], n)
	op += corr([ssz, ssz], [i, j], n)
	return op
end

function biquad(i::Int, j::Int, k::Int, l::Int, n::Int)::SparseOrFull{ComplexF64, Int64}
	# assumes operators commute
	# op =  corr([ssx, ssx, ssx, ssx], [i, j, k, l], n) 
	# op += corr([ssx, ssx, ssy, ssy], [i, j, k, l], n) 
	# op += corr([ssx, ssx, ssz, ssz], [i, j, k, l], n) 
	# op += corr([ssy, ssy, ssx, ssx], [i, j, k, l], n) 
	# op += corr([ssy, ssy, ssy, ssy], [i, j, k, l], n) 
	# op += corr([ssy, ssy, ssz, ssz], [i, j, k, l], n) 
	# op += corr([ssz, ssz, ssx, ssx], [i, j, k, l], n) 
	# op += corr([ssz, ssz, ssy, ssy], [i, j, k, l], n) 
	# op += corr([ssz, ssz, ssz, ssz], [i, j, k, l], n) 
	r = abs(k - l)
	op = dot_op(i, j, n) * chain_avg((x) -> dot_op(x, wrap_index(n, x + r), n), n)
	return op
end


"""
Basis of unique (subject to separation and ring geometry) pairs of correlators. 
This is a ⌊N/2⌋ - 1 dimensional space: we fix i = 1 then run from j = 2 to 
j = ⌊N/2⌋. Accounts for wrapping. Thus basis index i physically corresponds 
to distance on chain modulo ring structure. 
"""
function basis2spatial_idx(i::Int, n::Int)::Vector{Int}
	return [1, 1 + i]
end

function basis2spatial_idx_oc(i::Int, n::Int)::Vector{Int}
	return collect(combinations(1:n, 2))[i]
end

function op_in_basis(basis_idx, n)
	idx = basis2spatial_idx(basis_idx, n)
	r = idx[2] - idx[1]
	sum = dot_op(1, 1 + r, n)
	for i in 2:n
		sum += dot_op(wrap_index(n, i), wrap_index(n, i + r), n)
	end
	return (1 / n) * sum
end

function chain_avg(f, n)
	sum = f(1) 
	for i in 2:n
		sum += f(wrap_index(n, i))
	end 
	return (1 / n) * sum
end

function conn_corr2(i::Int, j::Int, n::Int, state::Vector{ComplexF64}; digits=14)::Float64
	i_spatial = basis2spatial_idx(i, n) 
	j_spatial = basis2spatial_idx(j, n) 

	oi = op_in_basis(i, n)

	r_i = abs(i_spatial[1] - i_spatial[2])
	r_j = abs(j_spatial[1] - j_spatial[2])
	bq = expect(oi * dot_op(1, 1 + r_j, n), state)

	# anticommutator 
	bq_ac = expect(dot_op(1, 1 + r_j, n) * oi, state)

	bq = bq + bq_ac

	sq = expect(dot_op(1, 1 + r_i, n), state) *
		 expect(dot_op(1, 1 + r_j, n), state) 

	return real(round((1/2) * bq - sq, digits=digits))
end

# Translationally averages both operators in biquadratic
function conn_corr(i::Int, j::Int, n::Int, state::Vector{ComplexF64})::Float64
	oi = op_in_basis(i, n)
	oj = op_in_basis(j, n)
	bq = expect(oi * oj, state)
	bq = bq + expect(oj * oi, state)
	sq = expect(oi, state) * expect(oj, state)

	# CM should be real so imaginary parts are due to floating errors; they mess
	# with diagonalization so we drop them.
	return real((1 / 2) * bq - sq)
end

# function conn_corr2(i::Int, j::Int, n::Int, state::Vector{ComplexF64})::ComplexF64
# 	i_spatial = basis2spatial_idx(i, n) 
# 	j_spatial = basis2spatial_idx(j, n) 

# 	bq = expect(biquad(i_spatial[1], i_spatial[2], j_spatial[1], j_spatial[2], n), 
# 		 state)
# 	sq = expect(dot_op(i_spatial[1], i_spatial[2], n), state) * 
# 		 expect(dot_op(j_spatial[1], j_spatial[2], n), state)
# 	return bq - sq
# end

function conn_corr_oc(i::Int, j::Int, n::Int, state::Vector{ComplexF64})::ComplexF64
	i_spatial = basis2spatial_idx_oc(i, n) 
	j_spatial = basis2spatial_idx_oc(j, n) 

	bq =  (1/2) * expect(biquad(i_spatial[1], i_spatial[2], j_spatial[1], j_spatial[2], n), state)
	# anticommutator 
	bq += (1/2) * expect(biquad(j_spatial[1], j_spatial[2], i_spatial[1], i_spatial[2], n), state)

	sq = expect(dot_op(i_spatial[1], i_spatial[2], n), state) *
	     expect(dot_op(j_spatial[1], j_spatial[2], n), state)
	return bq - sq
end

function hs_coeff(r::Int, n::Int, j_coupling::Real=1.0)::Float64
	d(r) = (n / π) * sin(r * π / n)
	return (j_coupling / (d(r)^2))
end

function hs_op(n::Int, j_coupling::Real=1.0)::SparseOrFull{ComplexF64, Int64}
	h = sparse(zeros(2^n, 2^n))

	for (i, j) in combinations(1:n, 2)
		h += hs_coeff(i - j, n, j_coupling) * dot_op(i, j, n)
	end

	return h 
end

function s_tot_op(n::Int)
	s = sparse(zeros(2^n, 2^n))
	for (i, j) in combinations(1:n, 2)
		s += dot_op(i, j, n)
	end
	return s
end

"""
Recall index of our operator basis corresponds to distance on chain
"""
function hs_coeffs(n::Int, j_coupling::Real=1.0)::Vector{Float64}
	return [hs_coeff(i, n, j_coupling) for i in 1:op_basis_dim(n)]
end

function hs_coeffs_oc(n::Int, j_coupling::Real=1.0)::Vector{Float64}
	return [begin k, l = basis2spatial_idx_oc(i, n); 
			hs_coeff(wrap_index(n, k, l), n, j_coupling) end 
			for i in 1:op_basis_dim_oc(n)]
end

function print_basis(n::Int)
	for i in 1:op_basis_dim(n)
		println(basis2spatial_idx(i, n))
	end
end

function make_corr_mat(n::Int, state::Vector{ComplexF64}; digits=14)::SparseOrFull{Float64, Int64}
	dim = op_basis_dim(n)
	return [conn_corr(i, j, n, state) for i in 1:dim, j in 1:dim]
end

# function make_corr_mat(n::Int, state::Vector{ComplexF64}; digits=14)::SparseOrFull{Float64, Int64}
# 	dim = op_basis_dim(n)
# 	return [conn_corr2(i, j, n, state, digits=digits) for i in 1:dim, j in 1:dim]
# end

function make_corr_mat_oc(n, state)
	dim = op_basis_dim_oc(n)
	return [conn_corr_oc(i, j, n, state) for i in 1:dim, j in 1:dim]
end
	
function in_nsp(nullsp, h, atol=1e-4)
	proj = 0 
	for v in nullsp
		proj += dot(h, v)^2
	end
	return proj, isapprox(proj, 1.0, atol=atol)
end

"""
dir = directory to save to
"""
function make_wf_cm(n::Int, dir; cm_digits=14)
	m = trunc(Int, n/2) # half filling
	print("\nMaking wavefunction... ")
	k = normalize(wf(m, n))
	print("Done.")
	print("\nMaking correlation matrix... ")
	cm = make_corr_mat(n, k, digits=cm_digits)
	print("Done.\n")
	save("$dir/cm_wf_n=$n.jld", "wf", k, "cm", cm)
	println("Saved to $dir/cm_wf_n=$n.jld")
	return Dict("wf" => k, "cm" => cm)
end

# assume we can sample just half of the ring - L -> L/2?
function ft_mat(n::Int) 
	return [exp(-1im * i * 2 * π * (j - 1) / n)  / n^(1/2) for i in 1:n, j in 1:n]
end 

# (1/3) * sparse(kron(sx, I(2), I(2)) * kron(I(2), sx, I(2)) + kron(sy, I(2), I(2)) * kron(I(2), sy, I(2)) + kron(sz, I(2), I(2)) * kron(I(2), sz, I(2)) 
# + kron(I(2), sx, I(2)) * kron(I(2), I(2), sx) + kron(I(2), sy, I(2)) * kron(I(2), I(2), sy) + kron(I(2), sz, I(2)) * kron(I(2), I(2), sz)
# + kron(I(2), I(2), sx) * kron(sx, I(2), I(2)) + kron(I(2), I(2), sy) * kron(sy, I(2), I(2)) + kron(I(2), I(2), sz) * kron(sz, I(2), I(2))) 

# op_in_basis(3, 4)