# Author: Alex Jacoby

import DelimitedFiles, LinearAlgebra


if isdir("Data") == true
    cd("Data")
else
    exit()
end

ogd = pwd()
datadir = readdir()


for fd in datadir
    cd(fd)
    mat = open("Correlation_Matrix.txt", "r") do io
        return DelimitedFiles.readdlm("Correlation_Matrix.txt")
    end
    ell = length(Vector(mat[1, :]))
    for j in 1:(ell-1)
        dn = string("Truncations=", j)
        mkdir(dn)
        cd(dn)
        touch("Eigvecs.txt")
        touch("Eigvals.txt")

        eigendat = LinearAlgebra.eigen(mat[1:(ell-j), 1:(ell-j)])

        eigendat.vectors
        eigendat.values
        open("Eigvecs.txt", "w") do io
            DelimitedFiles.writedlm("Eigvecs.txt", eigendat.vectors)
        end
        open("Eigvals.txt", "w") do io
            DelimitedFiles.writedlm("Eigvals.txt", eigendat.values)
        end
        cd(string(ogd,"/",fd))
    end
    cd(ogd)
end
