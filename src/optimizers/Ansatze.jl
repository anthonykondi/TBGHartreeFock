module Ansatze 
export P_ansatz, check_1rdm
using LinearAlgebra, StaticArrays, SparseArrays, Random
using ..KspGeoMapping


"""fill the first N_e diagonal components"""
function diag_fill(n::Int, N_e::Int)::Matrix{ComplexF64}
    A = zeros(n, n)
    for i in 1:N_e
        A[i, i] = 1.0 + im * 0.0
    end
    return A
end


"""get a random unitary matrix of size n x n"""
function random_unitary(n::Int, rng::AbstractRNG)::Matrix{ComplexF64}     
    Z = randn(rng, ComplexF64, n, n)
    Q, R = qr(Z)
    return Matrix(Q) * Diagonal(diag(R) ./ abs.(diag(R)))
end


function random_fill(n::Int, N_e, rng::AbstractRNG)::Matrix{ComplexF64}
    U = random_unitary(n, rng)
    A = diag_fill(n, N_e)
    return U * A * U'
end


"""n goes from 1 to N1xN2"""
function int_to_kidx(n::Int, sim::TBGSimulation)::SVector{2, Int}
    k2, k1 = divrem(n - 1, sim.moire_N[1])
    return SVector{2}(k1, k2)
end


"""
Obtain a 1rdm ansatz for specified filling, use type keyword to specify ansatz type.
"""
function P_ansatz(
    filling::Float64,
    sim::TBGSimulation;
    type::String = "random",
    rng::AbstractRNG = Random.default_rng()
)::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}

    N_e = floor(filling * sim.N_tot)
    P_dict = Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}()
    
    if type == "random"
        fill_dict = Dict(SVector{2}(kidx1, kidx2) => 0 for kidx1 in 0:(sim.moire_N[1] - 1), kidx2 in 0:(sim.moire_N[2] - 1))
        remaining = N_e
        while remaining > 0    # N_e electrons iid unif distributed among the N1xN2 k-points 
            bucket_int = rand(rng, 1:(sim.moire_N[1] * sim.moire_N[2]))
            bucket_k = int_to_kidx(bucket_int, sim)
            if fill_dict[bucket_k] == sim.geometry.N_tot    # check to see if that k point is already full
                continue
            end
            fill_dict[bucket_k] += 1
            remaining -= 1
        end
        for (k_idx, Nk_e) in fill_dict
            P_k = random_fill(sim.geometry.N_tot, Nk_e, rng)
            P_dict[SVector{2}(k_idx[1], k_idx[2])] = P_k
        end
    else
        error("Ansatz type not a valid type")
    end

    return P_dict
end


"""Print check of 1rdm properties"""
function check_1rdm(P::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}})     
    h_err = norm(P - P')
    p_err = norm(P - P * P)
    e_num = tr(P)

    println("h_err = $h_err")
    println("p_err = $p_err")
    println("e_num = $e_num")
end


end