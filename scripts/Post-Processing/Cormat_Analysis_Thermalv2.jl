# Author: Alex Jacoby

import DelimitedFiles, LinearAlgebra

if isdir("Thermal_Datav3") == true
    cd("Thermal_Datav3")
else
    exit()
end

ogd = pwd()
datadir = readdir()


for fd in datadir
    cd(fd)
    mat = open("Thermal_Correlation_Matrix.txt", "r") do io
        return DelimitedFiles.readdlm("Thermal_Correlation_Matrix.txt")
    end
    ell = length(Vector(mat[1, :]))
    for j in 0:(ell-1)
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
        cd(string(ogd, "/", fd))
    end
    cd(ogd)
end