# Author: Alex Jacoby

using LinearAlgebra
using SparseArrays
using Arpack

X = 1.0*[0 1; 1 0]
Y  = 1.0*[0 -im; im 0]
Z = 1.0*[1 0; 0 -1]
S = [X,Y,Z]


L = 15
Hdim = 2
id = zeros(Float64, Hdim, Hdim)+ I
zeromat = zeros(Float64,Hdim, Hdim)

## for on-site ops just fill out the first argument. It will, only read that far if you set op2 and pos2 to 0 (it will permit this)
function kronarray(op1::Matrix,op2::Matrix,pos1::Int64,pos2::Int64,Hdim::Int64,L::Int64)
    oparray = Vector{SparseMatrixCSC{ComplexF64, Int64}}([])
    for i in 1:L
        if (i != pos1) && (i != pos2)
            push!(oparray,sparse(id))
        elseif (i == pos1) || (i == pos2)
            if i == pos1
                push!(oparray,sparse(op1))
            elseif i == pos2
                push!(oparray,sparse(op2))
            end
        end
    end
    return oparray
end
##only works on 1:2L since we only need to use periodicity once.
function las(n::Int64,L::Int64)
    ls = 0
    ls = (n%(L+1))+(n÷(L+1))
    return ls
end
function dist(i::Int64,j::Int64,L::Int64)
    a = min(i,j)
    b = max(i,j)
    L1 = b-a
    L2 = a+L-b
    if L1>L2
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
function makeH(couplings::Matrix,Hdim::Int64,L::Int64)
    N = Hdim^L
    Hamiltonian = SparseMatrixCSC{ComplexF64,Int64}(zeros(ComplexF64,N,N))
    for i in 1:3, j in 1:L, k in j:L
        if j != k
            Hamiltonian = Hamiltonian + couplings[i,dist(j,k,L)]*sparse(kron(kronarray(S[i],S[i],j,k,Hdim,L)...))
        end
    end
    return Hamiltonian
end
function posops(t::Int64,d::Int64,Hdim::Int64,L::Int64)
    N = Hdim^L
    op = spzeros(ComplexF64,N,N)
    if d != L/2
        for i in 1:L
            op = op + kron(kronarray(S[t],S[t],las(i,L),las(i+d,L),Hdim,L)...)
        end
    elseif iseven(L) && (d == L÷2)
        for i in 1:(L÷2)
            op = op + kron(kronarray(S[t],S[t],las(i,L),las(i+d,L),Hdim,L)...)
        end
    end
    return op
end


function MakeOps(L::Int64,Hdim::Int64)
    Ops = []
    halfchain = L÷2
    for i in 1:halfchain
        op = posops(1,i,Hdim,L)+posops(2,i,Hdim,L)+posops(3,i,Hdim,L)
        push!(Ops,op)
    end
    return Ops
end

function cormat(Hdim::Int64,L::Int64)
    halfchain = L÷2
    cminit = zeros(ComplexF64,halfchain,halfchain)
    Ops = MakeOps(L,Hdim)
    OpsE = zeros(ComplexF64,length(Ops))
    for i in 1:length(Ops)
        OpsE[i] = ComplexF64((dpsi*Ops[i]*psi)[1,1])
    end
    for i in 1:halfchain, j in 1:i
        classex = OpsE[i]*OpsE[j]
        cminit[i,j] = (dpsi*Ops[i]*Ops[j]*psi)[1,1]-classex
        cminit[j,i] = cminit[i,j]
    end
    return cminit
end

function opdot(eigvec::Vector{Float64},Hdim::Int64,L::Int64)
    Ops = MakeOps(L,Hdim)
    oprecon = eigvec[1]*Ops[1]
    for i in 2:length(eigvec)
        oprecon = oprecon + eigvec[i]*Ops[i]
    end
    return oprecon
end


function momentum(n::Int64,L::Int64)
    return (2*pi*n)/L
end


function FT(L::Int64)
    n = L÷2
    FTmat = zeros(ComplexF64,n,n)
    for i in 1:n, j in 1:n
        FTmat[i,j] = exp(-im*i*momentum(j-1,n))/sqrt(n)
    end
    return FTmat
end

function cutoffmat(Λ::Int64,L::Int64)
    n = L÷2
    if Λ > n
        return false
    end
    cmat = zeros(Float64,n,Λ)
    for i in 1:Λ
        cmat[i,i] = 1.
    end
    return cmat
end

function Fourier(cm::Matrix{Float64},L::Int64)
    cmkspace = adjoint(FT(L))*cm*FT(L)
    return cmkspace
end


function Regulate(cmkspace::Matrix{ComplexF64},Λ::Int64,L::Int64)
    return transpose(cutoffmat(Λ,L))*cmkspace*cutoffmat(Λ,L)
end

function Regulated_CM(cm::Matrix{Float64},Λ::Int64,L::Int64)
    return Regulate(Fourier(cm,L),Λ,L)
end







## couplings reduced per the translational invariant basis construction
coup = zeros(Float64,3,L÷2)
# k=1
# coup[1,k] = coup[2,k] = coup[3,k] = 1.0
β = 2
for i in 1:(L÷2)
    #coup[:,i] = fill(rand(),3)
    #coup[:,i] = fill(exp(-i*β),3)
    coup[:,i] = fill((β/i)^2,3)
    # coup[:,i] = exp(-i*β)*rand(Float64,3)
end

H = makeH(coup,Hdim,L)
Ops = MakeOps(L,Hdim)
egdat = eigs(H,nev = 1,which = :SR,maxiter = 10^6)
psi = sparse(egdat[2])
dpsi = adjoint(psi)

hvec = normalize(coup[1,:])
ones = normalize(fill(1.,L÷2))
cmp = cormat(Hdim,L)
cm = real(cmp)
eigvals(cm)


k = L÷2-1
eigvals(transpose(cutoffmat(k,L))*cm*cutoffmat(k,L))
eigvecs(transpose(cutoffmat(k,L))*cm*cutoffmat(k,L))
dat = []
for k in 1:(L÷2)
    push!(dat,eigvals(transpose(cutoffmat(k,L))*cm*cutoffmat(k,L))[1])
    print(eigvals(transpose(cutoffmat(k,L))*cm*cutoffmat(k,L))[1])
end
dat
using Plots
plot(1:(L÷2),dat)






v1 =  eigvecs(cm)[:,1]
v2 = eigvecs(cm)[:,2]

c1 = dot(hvec,v1)
c2 = dot(hvec,v2)

q1 = dot(ones,v1)
q2 = dot(ones,v2)


q1^2+q2^2

Λ = 6
regcm = Regulated_CM(real(cormat(Hdim,L)),Λ,L)
eigvals(regcm)
eigvecs(regcm)[:,1]


hvec = normalize(coup[1,:])
khvec = adjoint(FT(L))*hvec
v1 = eigvecs(regcm)[:,1]
v2 = eigvecs(regcm)[:,2]
c1 = dot(khvec,v1)
c2 = dot(khvec,v2)
c1*v1+c2*v2+khvec

#eigvecs([1 0 0; 0 0 1; 0 1 0])[:,2]




using Plots
plot()
X1 = []
Y1 = []
evals = []
for i in 1:L
    push!(Y1,real(eigvals(cormatk(i,Hdim,L)))...)
    push!(X1,fill(Float64(i),3*i)...)
    push!(evals,real(eigvals(cormatk(i,Hdim,L))))
end


plot(X1,Y1,seriestype = :scatter)

evals



#savefig("CormatRG.pdf")


gaps = []
lowest = []
for i in 1:length(evals)
    push!(gaps,evals[i][2]-evals[i][1])
    push!(lowest, min(evals[i]...))
end
evals
gaps
plot(2:L,gaps[2:L])
plot(2:L,lowest[2:L])
