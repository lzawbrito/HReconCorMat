import LinearAlgebra
using DMRJtensor

# makes an initial MPS for spin s and size Ns
function makePsi0(s::Float64,Ns::Integer)

    n = Int64(2*s+1) # size of matrices
    initTensor = [zeros(ComplexF64,1,n,1) for i=1:Ns]
    for i = 1:Ns 
        #initTensor[i][1,1,1] = 1 # spin-up start
        for j = 1:n   # random start?
            initTensor[i][1,j,1] = rand(1)[1] + 1im*rand(1)[1]
        end
    end
    psi = MPS(ComplexF64,initTensor,1)
    return psi
end

# constructs the MPO for a given "power" of spin terms, e.g. power 3 -> \sum_{ijk} S_i S_j S_k
# NOT FINISHED
function construct_H_genericJ(H, H_op_vec, spin_mag::Float64, start_index::Integer, J_tensor)

    # H:           MPO reference (acts in place)
    # H_op_vec:    List of individual operators (acts in place)
    # spin_mag:    Spin magnitude
    # start_index: keeps track of where we are in the larger H MPO
    # J_tensor:    a nested list of depth (npowers - curr_power + 1), has the collection of J_terms we want for our model
    
    Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
    Sop_arr = [Sx,Sy,Sz] # spin operators
    nSpin = Int(2*spin_mag + 1) # dimension of spin space
    nterms = size(H,1) # total size of the MPO's onsite H
    
    op_idx_arr = [[1],[2],[3]] # p=1
    # op_idx_arr =  [[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]] # p=2
    # op_idx_arr =  [[1,1,1],[1,1,2],[1,1,3],[1,2,2],[1,2,3],[1,3,3],[2,2,2],[2,2,3],[2,3,3],[3,3,3]] # p=3
                   
    H_op_idx = 1
    for op_l_idx = 1:3 # loop over operators acting on left bond {x,y,z}
        for op_r_idx = 1:3 # loop over operators acting on right bond {x,y,z}

            lr_start = nterms - nSpin + 1 # index location of last row

            op_l_list = genOp(Sop_arr, op_idx_arr[op_l_idx])
            n_sym_l_ops = length(op_l_list)
            op_r_list = genOp(Sop_arr, op_idx_arr[op_r_idx])
            n_sym_r_ops = length(op_r_list)

            # make local operator
            size_H_op = 2 + (n_sym_l_ops)*(n_sym_r_ops)
            H_op_here = zeros(ComplexF64, size_H_op*nSpin, size_H_op*nSpin)
            H_op_here[             1:nSpin                           ,             1:nSpin            ] = Id # top left
            H_op_here[((size_H_op-1)*nSpin+1):size_H_op*nSpin, ((size_H_op-1)*nSpin+1):size_H_op*nSpin] = Id # bottom right
            op_idx_start = nSpin+1
            
            for sym_l_idx = 1:n_sym_l_ops
                for sym_r_idx = 1:n_sym_r_ops
                    
                    op_l = op_l_list[sym_l_idx]
                    op_r = op_r_list[sym_r_idx]
                    
                    # populate the Hamiltonian
                    idx_start = start_index + nSpin*(H_op_idx) # starting point along first column / last row
                    H[idx_start:(idx_start+nSpin-1), 1:nSpin                      ]  = J_tensor[op_l_idx,op_r_idx]*op_l # first column (has J!)
                    H[lr_start:(lr_start+nSpin-1),   idx_start:(idx_start+nSpin-1)] = op_r # last column (no J!)
                    H_op_idx += 1

                    # and populate the operator MPO
                    H_op_here[op_idx_start:(op_idx_start+nSpin-1)    ,                  1:nSpin             ] = op_l # first coulumn
                    H_op_here[((size_H_op-1)*nSpin+1):size_H_op*nSpin,   op_idx_start:(op_idx_start+nSpin-1)] = op_r # last row
                    op_idx_start += nSpin
                end
            end
        push!(H_op_vec,H_op_here) # push back into H_op_vec
        end
    end
        
end

# make all symmetry related operators for a given set of spin axes
function genOp(Sop_arr, op_idxs)
    # Sop_arr:    Array of the three spin operators, S_x, S_y, S_z
    # op_idx:     Array of the given spin axes
    N_terms = length(op_idxs)
    
    # TODO go over all unique permutations of op_idxs
    
    # return an array of spin operators correpsonding to each one
    #return op_list 
    
end

# constructs the MPO of a SU2 symmetry terms (e.g. one J per power)
function construct_H_SU2J(H, H_op_vec, spin_mag::Float64, start_index::Integer, J_array)

    # H:           MPO reference (acts in place)
    # H_op_vec:    List of individual operators (acts in place)
    # spin_mag:    Spin magnitude
    # start_index: keeps track of where we are in the larger H MPO (should usually be set to nSpin+1)
    # J_array:     array of length p, giving the couplings J_p
    
    Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
    Sop_arr = [Sx,Sy,Sz] # spin operators
    nSpin = Int(2*spin_mag + 1) # dimension of spin space
    max_p = nSpin - 1 # maximum allowed power of (S S)^p term
    nterms = size(H,1) # total size of the MPO's onsite H

    # size of each unique operator
    size_H_op = 2 + (3*max_p) # 3 terms per power, S_x^p, S_y^p, S_z^p
    
    lr_start = nterms - nSpin + 1 # index location of last row
    H_op_idx = 0 # keep track of which element of the Hamiltonian MPO we are on

    for p = 1:max_p # powers
        
        # each power yields a unique operator
        H_op_here = zeros(ComplexF64, size_H_op*nSpin, size_H_op*nSpin)
        H_op_here[             1:nSpin                           ,             1:nSpin            ] = Id # top left
        H_op_here[((size_H_op-1)*nSpin+1):size_H_op*nSpin, ((size_H_op-1)*nSpin+1):size_H_op*nSpin] = Id # bottom right
        op_idx_start = nSpin+1
        
        for dim = 1:3 # spin axes

            op_l = Sop_arr[dim]^p
            op_r = Sop_arr[dim]^p

            # populate the Hamiltonian
            idx_start = start_index + nSpin*(H_op_idx) # starting point along first column / last row
            H[idx_start:(idx_start+nSpin-1), 1:nSpin                      ]  = J_array[p]*op_l # first column (has J!)
            H[lr_start:(lr_start+nSpin-1),   idx_start:(idx_start+nSpin-1)] = op_r # last column (no J!)
            H_op_idx += 1

            # and populate the operator MPO
            H_op_here[op_idx_start:(op_idx_start+nSpin-1)    ,                  1:nSpin             ] = op_l # first coulumn
            H_op_here[((size_H_op-1)*nSpin+1):size_H_op*nSpin,   op_idx_start:(op_idx_start+nSpin-1)] = op_r # last row
            op_idx_start += nSpin
        end
        push!(H_op_vec,H_op_here) # push back into H_op_vec
    end
end

# construct the Hamiltonian MPO and operators for SU2 symmetric problem
function H_SU2(s::Float64, J_arr)
    # J_arr is an array of prefactors -> H = \sum_p J_p (S*S)^p
             
    Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
    n = Int(2*s + 1) # dimension of spin space
    max_p = n-1 # max allowed power of (S*S)^p terms
    nterms = n*(2 + 3*max_p) # the 2 keeps track of Id and O terms in MPO

    H_n = zeros(ComplexF64, nterms, nterms) # initialize
                     
    # set first column and last row identities
    H_n[1:n,1:n] = Id
    H_n[(nterms-n+1):nterms,(nterms-n+1):nterms] = Id
                        
    # we start in H just after the first [n x n] row block (or column block, if on last row)
    start_index = n+1;
    
    H_op_vec = [] # for keeping tack of each "local" H_operator matrix, will need to turn into an MPO afterwards!
    
    construct_H_SU2J(H_n, H_op_vec, s, start_index, J_arr) # iterative construction of (S*S)^p
    return H_n, H_op_vec
    
end
