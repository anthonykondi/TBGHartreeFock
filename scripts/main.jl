using TBGHartreeFock
using StaticArrays

# USE: julia --project=. scripts/main.jl

sim = TBGSimulation(1, 2, N1=2, N2=2)
P = P_ansatz(1/2, sim)
println("Ntot = $(sim.N_tot)")
println("Ntot = $(sim.geometry.N_tot)")

for (k, Pk) in P
    println("k = $k")
    println(check_1rdm(Pk))
end