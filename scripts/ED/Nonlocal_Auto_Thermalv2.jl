# Author: Alex Jacoby

##Loads all functions
import Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("KrylovKit")
Pkg.add("DelimitedFiles")

begin
    using LinearAlgebra, SparseArrays, KrylovKit, DelimitedFiles

    X = 1.0 * [0 1; 1 0]
    Y = 1.0 * [0 -im; im 0]
    Z = 1.0 * [1 0; 0 -1]
    S = [X, Y, Z]



    Hdim = 2
    id = zeros(Float64, Hdim, Hdim) + I
    zeromat = zeros(Float64, Hdim, Hdim)

    ## for on-site ops just fill out the first argument. It will, only read that far if you set op2 and pos2 to 0 (it will permit this)
    function kronarray(op1::Matrix, op2::Matrix, pos1::Int64, pos2::Int64, Hdim::Int64, L::Int64)
        oparray = Vector{SparseMatrixCSC{ComplexF64,Int64}}([])
        for i in 1:L
            if (i != pos1) && (i != pos2)
                push!(oparray, sparse(id))
            elseif (i == pos1) || (i == pos2)
                if i == pos1
                    push!(oparray, sparse(op1))
                elseif i == pos2
                    push!(oparray, sparse(op2))
                end
            end
        end
        return oparray
    end
    ##only works on 1:2L since we only need to use periodicity once.
    function las(n::Int64, L::Int64)
        ls = 0
        ls = (n % (L + 1)) + (n ÷ (L + 1))
        return ls
    end
    function dist(i::Int64, j::Int64, L::Int64)
        a = min(i, j)
        b = max(i, j)
        L1 = b - a
        L2 = a + L - b
        if L1 > L2
            return L2
        else
            return L1
        end
    end
    ##### This way is wrong, but it's subtle. Consult notes.
    # function makeH(couplings::Matrix,Hdim::Int64,L::Int64)
    #     N = Hdim^L
    #     Hamiltonian = SparseMatrixCSC{ComplexF64,Int64}(zeros(ComplexF64,N,N))
    #     for i in 1:3, j in 1:L, k in 1:(L-1)
    #         Hamiltonian = Hamiltonian + couplings[i,dist(j,las(j+k,L),L)]*sparse(kron(kronarray(S[i],S[i],j,las(j+k,L),Hdim,L)...))
    #     end
    #     return Hamiltonian
    # end
    ## This way is right!!!!!
    function makeH(couplings::Matrix, Hdim::Int64, L::Int64)
        N = Hdim^L
        Hamiltonian = SparseMatrixCSC{ComplexF64,Int64}(zeros(ComplexF64, N, N))
        for i in 1:3, j in 1:L, k in j:L
            if j != k
                Hamiltonian = Hamiltonian + couplings[i, dist(j, k, L)] * sparse(kron(kronarray(S[i], S[i], j, k, Hdim, L)...))
            end
        end
        return Hamiltonian
    end
    function posops(t::Int64, d::Int64, Hdim::Int64, L::Int64)
        N = Hdim^L
        op = spzeros(ComplexF64, N, N)
        if d != L / 2
            for i in 1:L
                op = op + kron(kronarray(S[t], S[t], las(i, L), las(i + d, L), Hdim, L)...)
            end
        elseif iseven(L) && (d == L ÷ 2)
            for i in 1:(L÷2)
                op = op + kron(kronarray(S[t], S[t], las(i, L), las(i + d, L), Hdim, L)...)
            end
        end
        return op
    end


    function MakeOps(L::Int64, Hdim::Int64)
        Ops = []
        halfchain = L ÷ 2
        for i in 1:halfchain
            op = posops(1, i, Hdim, L) + posops(2, i, Hdim, L) + posops(3, i, Hdim, L)
            push!(Ops, op)
        end
        return Ops
    end

    function cormat(Hdim::Int64, L::Int64, psi)
        dpsi = psi'
        halfchain = L ÷ 2
        cminit = zeros(ComplexF64, halfchain, halfchain)
        Ops = MakeOps(L, Hdim)
        OpsE = zeros(ComplexF64, length(Ops))
        for i in 1:length(Ops)
            OpsE[i] = ComplexF64((dpsi*Ops[i]*psi)[1, 1])
        end
        for i in 1:halfchain, j in 1:i
            classex = OpsE[i] * OpsE[j]
            cminit[i, j] = (dpsi*Ops[i]*Ops[j]*psi)[1, 1] - classex
            cminit[j, i] = cminit[i, j]
        end
        return cminit
    end
    function cormat_thermal(Hdim::Int64, L::Int64, psivec::Vector{Vector{ComplexF64}}, Boltzmann_weights::Vector{Float64}, Spectral_Depth::Int64)
        halfchain = L ÷ 2
        cminit = zeros(ComplexF64, halfchain, halfchain)
        Ops = MakeOps(L, Hdim)
        OpsE = zeros(ComplexF64, length(Ops))
        for i in 1:length(Ops), k in 1:Spectral_Depth
            OpsE[i] = OpsE[i] + ComplexF64(psivec[1]' * Ops[i] * psivec[1]) * Boltzmann_weights[k]
        end
        for i in 1:halfchain, j in 1:i
            Op1 = Ops[i] - LinearAlgebra.I * OpsE[i]
            Op2 = Ops[j] - LinearAlgebra.I * OpsE[j]
            for k in 1:Spectral_Depth
                cminit[i, j] = cminit[i, j] + 0.5 * (psivec[k]' * (Op1 * Op2 + Op2 * Op1) * psivec[k]) * Boltzmann_weights[k]
            end
            cminit[j, i] = cminit[i, j]
        end
        return real(cminit)
    end
    function opdot(eigvec::Vector{Float64}, Hdim::Int64, L::Int64)
        Ops = MakeOps(L, Hdim)
        oprecon = eigvec[1] * Ops[1]
        for i in 2:length(eigvec)
            oprecon = oprecon + eigvec[i] * Ops[i]
        end
        return oprecon
    end

    function do_everything_please(L, δ, Spectral_Depth, β)
        coup = zeros(Float64, L ÷ 2)
        for i in 1:(L÷2)
            coup[i] = 1 / i^δ
        end
        Ops = MakeOps(L, Hdim)
        hvec = normalize(coup)
        #H = makeH(coup,Hdim,L)
        H = opdot(hvec, Hdim, L)
        H = H - (LinearAlgebra.tr(H) / (2^L)) * LinearAlgebra.I

        egdat = eigsolve(H, Spectral_Depth, :SR, Float64)
        pre_spec = egdat[1][1:Spectral_Depth]
        inverse_thermal_scale = abs(pre_spec[1]-pre_spec[2])^-1
        spectrum = inverse_thermal_scale * pre_spec
        Boltzmann_weights = exp.(-β * spectrum) / (sum(exp.(-β * spectrum)))

        #test to make sure the indexing is right
        # Boltzmann_weights[1] #largest Boltzmann weight
        # psivec[1]' * H * psivec[1] #should be lowest energy

        psivec = egdat[2][1:Spectral_Depth]
        cmp = cormat_thermal(Hdim, L, psivec,Boltzmann_weights,Spectral_Depth)
        return (cmp, hvec, Boltzmann_weights, egdat)
    end
    # function momentum(n::Int64, L::Int64)
    #     return (2 * pi * n) / L
    # end


    # function FT(L::Int64)
    #     n = L ÷ 2
    #     FTmat = zeros(ComplexF64, n, n)
    #     for i in 1:n, j in 1:n
    #         FTmat[i, j] = exp(-im * i * momentum(j - 1, n)) / sqrt(n)
    #     end
    #     return FTmat
    # end

    # function cutoffmat(Λ::Int64, L::Int64)
    #     n = L ÷ 2
    #     if Λ > n
    #         return false
    #     end
    #     cmat = zeros(Float64, n, Λ)
    #     for i in 1:Λ
    #         cmat[i, i] = 1.0
    #     end
    #     return cmat
    # end

    # function Fourier(cm::Matrix{Float64}, L::Int64)
    #     cmkspace = adjoint(FT(L)) * cm * FT(L)
    #     return cmkspace
    # end


    # function Regulate(cmkspace::Matrix{ComplexF64}, Λ::Int64, L::Int64)
    #     return transpose(cutoffmat(Λ, L)) * cmkspace * cutoffmat(Λ, L)
    # end

    # function Regulated_CM(cm::Matrix{Float64}, Λ::Int64, L::Int64)
    #     return Regulate(Fourier(cm, L), Λ, L)
    # end
end


#NOTE THE WAY YOUR TEMPERATURE IS SET UP IS EXPONENTIALLY SENSITIVE TO SYSTEM SIZE-- ie finite temperature effects become exponentially stronger in system size. To get rid of this problem, you would need to make your temperature scale the gap rather than the full spectral width. Will need to implement this as well.


L_trials = 18:2:18
δ_trials = 2:2:2
Spectral_Depth = 20
β_range = 0.5:0.5:4.5

ogd = pwd()
if !(isdir("Thermal_Datav3"))
    mkdir("Thermal_Datav3")
end
cd(string(ogd, "/Thermal_Datav3"))




for L in L_trials, δ in δ_trials, β in β_range
    filename = string("Cormat_Data_L=", L, "_Delta=", δ, "_Spectral_Depth=", Spectral_Depth, "_beta=", β)
    mkdir(filename)
    cd(filename)
    mkdir("Extra_Data")
    A = string("Thermal_Correlation_Matrix.txt")
    B = string("Extra_Data/Hvec.txt")
    C = string("Extra_Data/Boltzmann_weights.txt")
    D = string("Extra_Data/Raw_Spectrum.txt")
    touch(A)
    touch(B)
    touch(C)
    touch(D)


    all_stuff = do_everything_please(L, δ, Spectral_Depth, β)


    open(A, "w") do io
        DelimitedFiles.writedlm(A, all_stuff[1])
    end
    open(B, "w") do io
        DelimitedFiles.writedlm(B, all_stuff[2])
    end
    open(C, "w") do io
        DelimitedFiles.writedlm(C, all_stuff[3])
    end

    open(D, "w") do io
        DelimitedFiles.writedlm(D, all_stuff[4][1][1:Spectral_Depth])
    end

    cd(string(ogd, "/Thermal_Datav3"))
    print(string("done_", filename))
end






