using TBGHartreeFock
using StaticArrays

# USE: julia --project=. scripts/main.jl

sim = TBGSimulation(1, 2)
plot_thresh_sites(sim, N_moire=SVector(1, 1))