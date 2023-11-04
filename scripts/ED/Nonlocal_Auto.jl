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

    function opdot(eigvec::Vector{Float64}, Hdim::Int64, L::Int64)
        Ops = MakeOps(L, Hdim)
        oprecon = eigvec[1] * Ops[1]
        for i in 2:length(eigvec)
            oprecon = oprecon + eigvec[i] * Ops[i]
        end
        return oprecon
    end


    function momentum(n::Int64, L::Int64)
        return (2 * pi * n) / L
    end


    function FT(L::Int64)
        n = L ÷ 2
        FTmat = zeros(ComplexF64, n, n)
        for i in 1:n, j in 1:n
            FTmat[i, j] = exp(-im * i * momentum(j - 1, n)) / sqrt(n)
        end
        return FTmat
    end

    function cutoffmat(Λ::Int64, L::Int64)
        n = L ÷ 2
        if Λ > n
            return false
        end
        cmat = zeros(Float64, n, Λ)
        for i in 1:Λ
            cmat[i, i] = 1.0
        end
        return cmat
    end

    function Fourier(cm::Matrix{Float64}, L::Int64)
        cmkspace = adjoint(FT(L)) * cm * FT(L)
        return cmkspace
    end


    function Regulate(cmkspace::Matrix{ComplexF64}, Λ::Int64, L::Int64)
        return transpose(cutoffmat(Λ, L)) * cmkspace * cutoffmat(Λ, L)
    end

    function Regulated_CM(cm::Matrix{Float64}, Λ::Int64, L::Int64)
        return Regulate(Fourier(cm, L), Λ, L)
    end
end



ogd = pwd()
if !(isdir("Data"))
    mkdir("Data")
end




for j in 2:10
    for i in 1:6
        cd(string(ogd, "/Data"))
        L = 2 * j
        δ = i

        filename = string("Cormat_Data_L=", L, "_Delta=", δ)
        mkdir(filename)
        cd(filename)
        ## couplings reduced per the translational invariant basis construction
        coup = zeros(Float64, L ÷ 2)
        for i in 1:(L÷2)
            coup[i] = 1 / i^δ
        end
        Ops = MakeOps(L, Hdim)
        hvec = normalize(coup)
        #H = makeH(coup,Hdim,L)
        H = opdot(hvec, Hdim, L)





        egdat = eigsolve(H, 1, :SR, Float64)
        psi = sparse(normalize(egdat[2][1]))


        cmp = cormat(Hdim, L,psi)
        cm = real(cmp)


        A = string("Correlation_Matrix.txt")
        touch(A)
        open(A, "w") do io
            DelimitedFiles.writedlm(A, cm)
        end
        cd(ogd)
        print(string("done_delta=",i,"+L=",2*j,"         "))
    end
end