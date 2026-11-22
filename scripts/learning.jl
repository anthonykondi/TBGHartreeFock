# using Distributed, ChunkSplitters, Random, LinearAlgebra, TBGHartreeFock

# function main()
#     matrices = [rand(500, 500) for _ in 1:100]
#     spectrum = []
#     maxiter = length(matrices)
#     iter = 0
#     @time for matrix in matrices
#         iter += 1
#         F = eigen(matrix)
#         push!(spectrum, F.values)
#         progress("Matrices diagonalized", iter, maxiter)
#     end
#     println("DONE")
# end

# =====================================================================

using Distributed

addprocs(5; exeflags="--project")   # workers use the same project environment

@everywhere begin
    using LinearAlgebra
    BLAS.set_num_threads(1)         # avoid oversubscribing cores (see below)
end

using Random, TBGHartreeFock        # only needed on the main process (for `progress`)

function main()
    matrices = [rand(2000, 2000) for _ in 1:50]
    maxiter  = length(matrices)

    # workers can't print to your progress bar directly, so they report through a channel
    progress_ch = RemoteChannel(() -> Channel{Int}(maxiter))
    reporter = @async for iter in 1:maxiter
        take!(progress_ch)
        progress("Matrices diagonalized", iter, maxiter)
    end

    @time spectrum = pmap(matrices) do matrix
        vals = eigen(matrix).values
        put!(progress_ch, 1)
        return vals
    end

    wait(reporter)
    println("DONE")
    return spectrum
end

spectrum = main()