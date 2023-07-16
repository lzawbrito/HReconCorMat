using LinearAlgebra, Statistics
#using GLMakie, CairoMakie, LaTeXStrings
using CairoMakie, LaTeXStrings
using DelimitedFiles

CairoMakie.activate!(type="png")
#GLMakie.activate!()
#Makie.inline!(true)

function addJ(J,R,H)
    N = size(H,1)
    dH = diagm([J for i = 1:N])
    dH = circshift(dH,-R)
    H = H + dH + dH'
end

H = function randJ(J,R,N,σ)
    H = zeros(N,N)
    #rand_ratio = 0.5;
    for i = 1:N
        j = mod(Int(i+R)-1,N) + 1
        H[i,j] = J*((1-σ) + 2*σ*rand())
    end
    H = H + H'
    return H
end

function rangeTruncationCorMat(N, σ, R_list, J_list)

    nOps = length(R_list)

    op_list = [randJ(J_list[idx], R_list[idx], N, σ) for idx = 1:nOps]
    H = sum(op_list)

    vals, vecs = eigen(H)
    ψ0 = vecs[:,1]

    op_expects = [ψ0'*op_list[idx]*ψ0 for idx = 1:nOps]

    corMat = zeros(nOps,nOps)
    for i = 1:nOps
        Oi = op_list[i]
        for j = 1:nOps
            Oj = op_list[j]
            corMat[i,j] = 0.5*ψ0'*(Oi*Oj + Oj*Oi)*ψ0 - (op_expects[i]*op_expects[j])
            
        end
    end

    λ = zeros(nOps)
    for idx = 1:nOps
        c_vals,c_vecs = eigen(corMat[1:idx,1:idx])
        λ[idx] = c_vals[1]
    end

    return λ

end

N = 50;
σ = 0.5;
R_list = 1:(N/2-1)
J_list_array = zeros(4,length(R_list))
J_list_array[1,:] = 1 ./ R_list.^2
J_list_array[2,:] = 1 ./ R_list.^3
J_list_array[3,:] = exp.(-R_list/2)
J_list_array[4,:] = exp.(-(R_list/7).^2)
J_list_array = J_list_array ./ J_list_array[:,1]

J_labels = [L"$1/R^2$",L"$1/R^3$",L"$e^{-R/2}$",L"$e^{-(R/8)^2}$"]

writedlm("./data/stephen_data/tb_j_labels.txt", J_labels)

fig = Figure(resolution=(600,800))
ax1 = Axis(fig[1,1], yscale=log10)
ax2 = Axis(fig[2,1], yscale=log10)

writedlm("./data/stephen_data/tb_j.txt", J_list_array)

lambda_means = zeros(size(J_list_array)[1], size(J_list_array)[2])
lambda_stds = zeros(size(lambda_means)...)

for idx = 1:size(J_list_array,1)
    
    J_list = J_list_array[idx,:]
    N_samps = 50;
    λ = zeros(N_samps,length(R_list))
    for idx = 1:N_samps
        λ[idx,:] = rangeTruncationCorMat(N, σ, R_list, J_list)
    end
    λ_m = mean(λ,dims=1)
    λ_s = std(λ,dims=1)
    λ_high = λ_m + λ_s
    λ_low = λ_m - λ_s

    lambda_means[idx, :] = λ_m
    lambda_stds[idx, :] = λ_s

    # J plot
    Jh_rescale = λ_m[end-1]/J_list[end-1].^2
    lh = lines!(ax1,R_list,Jh_rescale*J_list.^2,linestyle=:dash)#,label=J_labels[idx])

    # λ plot
    lh = lines!(ax1,R_list[1:end-1], abs.(λ_m[1:end-1]),#./J_list[1:end-1].^2,
        label=J_labels[idx], color=lh.color)
    # b_color = RGBAf0(RGBf0(lh.color[]),0.3)
    band!(ax1,R_list[1:end-1], abs.(λ_low[1:end-1]),#./J_list[1:end-1].^2, 
                              abs.(λ_high[1:end-1]),#./J_list[1:end-1].^2,
            # color = b_color)
    )

                # λ plot
    lines!(ax2,R_list[1:end-1], abs.(λ_m[1:end-1])./J_list[1:end-1].^2,
    label=J_labels[idx],color=lh.color)
    # b_color = RGBAf0(RGBf0(lh.color[]),0.3)
    band!(ax2,R_list[1:end-1], abs.(λ_low[1:end-1])./J_list[1:end-1].^2, 
                          abs.(λ_high[1:end-1])./J_list[1:end-1].^2,)
        # color = b_color)

end

writedlm("./data/stephen_data/tb_lambda_means.txt", lambda_means)
writedlm("./data/stephen_data/tb_lambda_stds.txt", lambda_stds)



# axislegend(ax1)
# axislegend(ax2)

# text!(ax1,L"(a)",position=Point2f0(1,10),textsize=25)
# xlims!(ax1,0,25)
# ylims!(ax1,1e-13,1e3)

# text!(ax2,L"(b)",position=Point2f0(1,2),textsize=25)
# xlims!(ax2,0,25)
# ylims!(ax2,5e-4,7)

# ax1.ylabel="λ(R)"
# ax2.xlabel="R"
# ax2.ylabel="Q(R)"
# display(fig)

#CairoMakie.activate!()
#save("cormat_lambdas.pdf",fig)
#save("cormat_Qs.pdf",fig)
save("./fig/cormat_Jsq_fit.pdf",fig)

