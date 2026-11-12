module KspSCFOptimizers
export SCFOutput, SCF_optimize
using LinearAlgebra, StaticArrays, SparseArrays, Distributed, ChunkSplitters, FeastKit, BenchmarkTools
using ..KspGeoMapping, ..Ansatze


struct SCFOutput
    H_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    E_history::Vector{Float64}
    total_iters::Int
end


"""Get the converged HF hamiltonian (filling is inferred from trace(P_0))"""
function SCF_optimize(
    P_ansatz_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    T_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation;
    ϵ_tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    kT::Float64 = 0.0,
    parallelized::Bool = false     # keep to false when running on a cluster
)::SCFOutput

    N_e = Int(tr())

end


"""
Get the converged HF hamiltonian (specify filling and P_0 ansatz type).
Builds T_dict and P_ansatze_dict in here (wrapper function) and then passes them on to the main function.
"""
function SCF_optimize(
    filling::Float64,
    type::String,
    sim::TBGSimulation;
    ϵ_tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    kT::Float64 = 0.0,
    parallelized::Bool = false 
)::SCFOutput

    # FILL IN

end



end