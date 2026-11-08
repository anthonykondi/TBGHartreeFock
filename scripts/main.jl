using TBGHartreeFock
using StaticArrays

# USE: julia --project=. scripts/main.jl

sim = TBGSimulation(4, 5)
plot_thresh_sites(sim, N_moire=SVector(1, 1))
plot_TB_band_structure(sim)