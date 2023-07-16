#=
This script includes miscellaneous tests of certain properties of the 
correlators. 

Author: Lucas Z. Brito
=#

using DataFrames
using CSV
using LinearAlgebra
using PyPlot
using JSON
using Arpack

include("./src/hs_tools.jl")

output_dir = "../data/hs/" 

"""
Compute the "classical" correlation <S_1 * S_2> <S_{r+2} * S_{r+3}>, the
biquadratic term <(S_1 * S_2)(S_{r+1} * S_{r+}), and the the connected 
correlator (biquadratic - conn).
"""
function conn_corr_vary_r(n::UInt8)
	m = UInt8(trunc(n / 2))
	println("Evaluating wavefunction for n = $n, m = $m...")

	k = normalize(wf(m, n))
	rs = 1:(m - 1)

	println("Evaluating classical correlation terms...")
	cl = [real(expect(dot_op(1, 2, n), k) * expect(dot_op(r + 2, r + 3, n), k))
		for r in rs] 

	println("Evaluating biquadratic term...")
	bq = [real(expect(biquad(1, 2, r + 2, r + 3, n), k)) for r in rs]

	println("Evaluating connected correlator...")
	conn = [real(conn_corr(1, 2, r + 2, r + 3, n, k)) for r in rs]

	println("Constructing DataFrame...")
	df = DataFrame(r=rs, bq=bq, cl=cl, conn=conn)

	println("Writing to file...")
	fn = CSV.write(output_dir * "hs_corr_n=$n.csv", df)
	println("Successfully written to " * fn)
end

"""
Perform reconstruction of the Haldane Shastry Hamiltonian and write data to a 
CSV file. 
"""
function reconstruct(n)
	m = Int(trunc(n / 2))
	println("Evaluating wavefunction...")
	k = normalize(wf(m, n))
	
	println("Calculating HS coefficients...")
	hs = hs_coeffs(n)
	
	println("Making correlation matrix...")
	cm = make_corr_mat(n, k)

	println("Diagonalizing correlation matrix...")
	eig = eigen(cm)
	vecs = round.(eig.vectors, digits=4)
	vals = round.(eig.values, digits=4)
	
	# collect nonzero 
	first_nonzero = findfirst(x -> x != 0, vals)
	
	## solve for linear combination of zero eigenvectors that produces HS? 
	a = vecs[:, 1:(first_nonzero - 1)]
	coeffs = a \ hs
	recon = a * coeffs

	d = Dict("hs" => hs, 
			 "coeffs" => coeffs, 
			 "cm" => cm, 
			 "evals" => vals, 
			 "evecs" => vecs, 
			 "recon" => recon)

	open(output_dir * "hs_recon_n=$n.json", "w") do f 
		JSON.print(f, d)
	end

end

"""
Test to see if Gutzwiller wavefunction is indeed eigenvector of HS Hamiltonian 
operator.
"""
function test_wf(n)
	m = Int(trunc(n / 2))
	k = normalize(wf(m, n))
	hs = hs_op(n, -1)
	hs_k = hs * k
	factor = (idx = findfirst(x -> x != 0, k); k[idx] / hs_k[idx])
	return k, hs_k, hs_k * factor
end

"""
Verifies that the correlator <S_i * S_j> is indeed translationally invariant. 
"""
function verify_corr_trans_inv(dr, wf, n)
	is, js, exps = [], [], []
	for idx in n/2:(n/2 + n)
		i = Int(idx) % n + 1
		j = (i + dr) % n + 1
		append!(exps, round(expect(dot_op(i, j, n), wf), digits=10))
		append!(is, i)
		append!(js, j)
	end

	df = DataFrame(i=is, j=js, dot_corr_re=real.(exps), dot_corr_im=imag.(exps))
	fn = CSV.write(output_dir * "dot_corr_trans_n=$(n)_r=$dr.csv", df)
	println("Written to $fn.")
end

"""
Check that the Hamiltonian obeys the ring "reflexivity" property.

If we are working with a ring, wrapped coefficient and i - j coefficient 
should be the same. If they are, we only have to go up to \Delta t = N / 2; past 
that we can simply "reflect" the operators along the ring's diameter.
"""
function h_reflect_check(n, atol=1e-10)
	printstyled("N = $n REFLEXIVITY CHECK\n\n", bold=true, underline=true)
	println("n/2 = $(Int(trunc(n/2)))")
	println("atol = $atol")
	println()
	i = 1
	for j in 2:n
		println("i = $i, j = $j")
		println("i - j = \t$(i - j)")
		wrap = wrap_index(n, i, j)
		println("wrap(i - j) = \t$(wrap)")

		wrap_coeff = hs_coeff(wrap, n)
		unwrap_coeff = hs_coeff(i - j, n)
		
		println("wrapped = \t$(round(wrap_coeff, digits=5))")
		println("unwrap = \t$(round(unwrap_coeff, digits=5))")
		if isapprox(wrap_coeff, unwrap_coeff, atol=atol)
			printstyled("Match\n", color=:green, bold=true)
		else
			printstyled("Not equal\n", color = :red, bold=true)
		end
		println()
	end
end

"""
Check that the wavefunction obeys the ring "reflexivity" property (see docstring 
for `h_reflect_check`).
"""
function wf_reflect_check(n, state, atol=1e-10)
	printstyled("N = $n REFLEXIVITY CHECK\n\n", bold=true, underline=true)
	println("n/2 = $(Int(trunc(n/2)))")
	println("atol = $atol")
	println()
	j = 1
	for i in 2:n
		println("i = $i, j = $j")
		println("i - j = \t$(i - j)")
		wrap = n - abs(1 - i)
		println("wrap(i - j) = \t$(wrap)")

		wrap_corr = expect(dot_op(i, j, n), state)
		unwrap_corr = expect(dot_op(j, wrap + j, n), state)
		
		println("wrapped = \t$(round(wrap_corr, digits=5))")
		println("unwrap = \t$(round(unwrap_corr, digits=5))")
		if isapprox(wrap_corr, unwrap_corr, atol=atol)
			printstyled("Match\n", color=:green, bold=true)
		else
			printstyled("Not equal\n", color = :red, bold=true)
		end
		println()
	end
end

"""
Check that the biquadratic expectation value <(S_i * S_j)(S_j * S_k)> obeys the
ring "reflexivity" property (see docstring for `h_reflect_check`).
"""
function bq_reflect_check(n, state, atol=1e-10)
	printstyled("N = $n REFLEXIVITY CHECK\n\n", bold=true, underline=true)
	println("n/2 = $(Int(trunc(n/2)))")
	println("atol = $atol")
	println()
	j = 1
	for i in 3:(n - 1)
		println("i = $i, j = $j")
		println("i - j = \t$(i - j)")
		wrap = n - abs(1 - i)
		println("wrap(i - j) = \t$(wrap)")

		wrap_corr = expect(biquad(j, i, j + 1, i + 1, n), state)
		unwrap_corr = expect(biquad(j, wrap + j, j + 1, wrap + j + 1, n), state)
		
		println("wrapped = \t$(round(wrap_corr, digits=5))")
		println("unwrap = \t$(round(unwrap_corr, digits=5))")
		if isapprox(wrap_corr, unwrap_corr, atol=atol)
			printstyled("Match\n", color=:green, bold=true)
		else
			printstyled("Not equal\n", color = :red, bold=true)
		end
	end
end

"""
Check that the correlation matrix obeys the ring "reflexivity" property (see
docstring for `h_reflect_check`).
"""
function corr_mat_reflect_check(n, state, atol=1e-10)
	for i in 1:(n - 1) 
		println(basis2spatial_idx(i, n))
		println(real(corr_mat_entry(1, i, n, state)))
	end
end

"""
Check that the biquadratic triplet and singlet Casimir invariant expectation 
values are the correct (analytically determined) values.
"""
function test_casimir()
	η = 2
	trip = kron(up, up)
	sing = normalize(kron(up, down) - kron(down, up))
	
	bq = biquad(1, 2, 1, 2, η)
	bl = dot_op(1, 2, η)
	
	expect(bl, trip)
	expect(bl, sing)
	bq_trip = expect(bq, trip) 
	bq_sing = expect(bq, sing)
	println(isapprox(bq_trip, 1/16, atol=1e-10))
	println(isapprox(bq_sing, 9/16, atol=1e-10))
end


##### RECONSTRUCTION ###########################################################
printstyled("----------- RECONSTRUCTING ------------\n\n", bold=true)
n = 15
atol = 1e-2
make_wf_cm(n, "./hs/data/")
# m = Int(trunc(n / 2))
# printstyled("Evaluating wavefunction...\n", bold=true)
# k = normalize(wf(m, n))
# printstyled("Calculating HS coefficients...\n", bold=true)
hs = normalize(hs_coeffs(n))

printstyled("\nHS coefficients:\n", bold=true)
display(hs); println()


printstyled("Making correlation matrix...\n", bold=true)
cm = make_corr_mat(n, k)
display(cm)
re_cm = real(cm)

cm2 = make_corr_mat2(n, k, digits=15)
re_cm2 = real(cm2)
re_cm2[1, 1]
re_cm[1,  1]

# re_cm2 = round.(re_cm2, digits=14)
re_cm - re_cm2
println()
printstyled("Diagonalizing correlation matrix...\n\n", bold=true)
eig = eigen(re_cm2)

printstyled("Eigenvectors:\n", bold=true)
display(eig.vectors); print("\n")
printstyled("Eigenvalues:\n", bold=true)
display(eig.values); print("\n")

nullsp = [normalize(eig.vectors[:, i]) for i in 1:2]
proj, is_in_nsp = in_nsp(nullsp, hs, atol)
in_nsp([eig.vectors[:, 1]], hs, atol)

conserved_quant = normalize(ones(trunc(Int, n / 2)))
in_nsp(nullsp, conserved_quant, atol)

coeffs_recon = reduce((a, b) -> a + dot(b, hs) * normalize(b), nullsp; init=zeros(trunc(Int, n / 2)))

if is_in_nsp
	printstyled("In nullspace", bold=true, color=:green)
else 
	printstyled("Not in nullspace", bold=true, color=:red)
end 

printstyled(" (atol = $atol)\n", bold=true)
printstyled("Projection: ", bold=true)
print(proj)
println()


begin
	df = DataFrame(r=1:(trunc(Int, n / 2)), 
				   n=n,
				   coeffs_anal=hs, 
				   coeffs_recon=real.(coeffs_recon))
	fn = CSV.write(output_dir * "recon_n=$n.csv", df)
	printstyled("Written reconstruction data to $fn", bold=true, color=:green)
end


# e = 9
# r = 1
# sub_op = zeros((2^n, 2^n))
# for c in eig.vectors[:, e]
# 	sub_op += c * dot_op(1, r + 1, n)
# 	r += 1
# end
# expect(sub_op * sub_op, k) - expect(sub_op, k) ^ 2
# vals[e]
