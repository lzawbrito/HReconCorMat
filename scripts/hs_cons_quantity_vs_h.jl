#= 
This script studies the two eigenvectors of the CM with lowest eigenvalues and 
projects them onto the S_tot operator and H. 

Author: Lucas Z. Brito
=#

using LinearAlgebra 
using Plots
using JLD
using DataFrames
using CSV

include("./src/hs_tools.jl")
data_dir = "./data/hs_data/"

function trunc_mat(a, n)
	return a[1:end - n, 1:end - n]
end

function proj(a, b)
	# Let b be the shorter vector
	a, b = length(a) < length(b) ? (a, b) : (b, a)
	b_tr = b[1:length(a)]

	return dot(a, b_tr) / (norm(a) * norm(b_tr))
end

function bias(v, v_1, v_2) 
	proj1 = proj(v, v_1)
	proj2 = proj(v, v_2)
	return proj1 / (proj1 + proj2), proj2 / (proj1 + proj2)
end

l = 5

# Check that H and S_tot commute. Explicitly constructing the operators is
# quite costly so only run this for lower values of l
if l < 10
	println("Checking that H and S_tot commute...")
	h = hs_op(l) 
	s = s_tot_op(l)
	comm = h * s - s * h
	
	printstyled("Maximum values (abs) of commutator:\n", bold=true)
	println(maximum(abs.((Matrix(comm)))))
	println()
end

println("Loading data for l=$l...")
data = load(data_dir * "cm_wf_n=$l.jld")

cm = data["cm"]

full_eig = eigen(cm)
n = size(cm)[1]

# full_eig.vectors[:, 1]
h = hs_coeffs(l)
cons = normalize(ones(n))

# Compute eigenvalues and biases
println("Computing eigenvalues and biases...")
eval1 = zeros(n - 1)
eval2 = zeros(n - 1)
bias1 = zeros(n - 1)
bias2 = zeros(n - 1)
for i in 0:(n - 2)
	trunc_cm = trunc_mat(cm, i)

	eig = eigen(trunc_cm)

	evals = eig.values[1:2]
	evecs = eig.vectors[:, 1:2]

	eval1[i + 1] = evals[1]
	eval2[i + 1] = evals[2]

	bias1[i + 1] = begin
		b1 = bias(evecs[:, 1], h, cons)
		b1[1] - b1[2]
	end 

	bias2[i + 1] = begin
		b2 = bias(evecs[:, 2], h, cons)
		b2[1] - b2[2]
	end 
end

# Write data to CSV 
println("Writing to CSV...")
df = DataFrame(eval1=eval1,
			   eval2=eval2, 
			   bias1=bias1, 
			   bias2=bias2)

fn = data_dir * "hs_cons_quantity_vs_h.n=$l.csv"
CSV.write(fn, df)
println("Written to $fn.")