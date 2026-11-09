module KspHamiltonians
export construct_T_k, construct_Hartree_k, construct_Fock_k, get_HF_hamiltonian_k

using LinearAlgebra, StaticArrays, SparseArrays 

using ..KspGeoMapping


"""
Construct T_k sparse matrix given k reciprocal lattice vector. (this works for any k)! 
"""
function construct_T_k(
    k::SVector{3, Float64}, 
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    dim = 8 * (sim.geometry.mn[1]^2 + sim.geometry.mn[1] * sim.geometry.mn[2] + sim.geometry.mn[2]^2)    # needed for fixing sparse matrix dimensions
    srows = Vector{Int}()
    scols = Vector{Int}()
    selements = Vector{ComplexF64}()
    for (mi_μ, coupled_mi_v) in sim.coupled_midxs    # loop through all matrix indices (1st)

        for mi_v in coupled_mi_v    # loop through all connected matrix indices (2nd)
            connections = sim.coupled_midxs_tunneling[SVector{2}(mi_μ, mi_v)]
            matrix_element = 0.
            for (η, _, t_μvη) in connections    # loop through all connected moire cells
                matrix_element += t_μvη * exp(im * dot(k, η))
            end

            if isapprox(matrix_element, 0, atol=1e-8)
                continue
            end

            push!(srows, mi_μ)
            push!(scols, mi_v)
            push!(selements, matrix_element)
        end
    end

    T_k = sparse(srows, scols, selements, dim, dim)
    return T_k
end


"""Construct T_k sparse matrix given k reciprocal lattice vector integer coordinates"""
function construct_T_k(k_idx::SVector{2, Int}, sim::TBGSimulation)::SparseMatrixCSC{ComplexF64}
    k = frac_to_cart(k_idx[1] / sim.moire_N[1], k_idx[2] / sim.moire_N[2], sim.geometry)
    return construct_T_k(k, sim)
end


"""
Construct Hartree sparse matrix density matrix dictionary P_dict (keys are k integer coordinates)
Note that the Hartree mean field hamiltonian does not depend on k-point (so we don't put k in the call signature)
"""
function construct_Hartree_k(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    dim = 8 * (sim.geometry.mn[1]^2 + sim.geometry.mn[1] * sim.geometry.mn[2] + sim.geometry.mn[2]^2)    # needed for fixing sparse matrix dimensions
    srows = Vector{Int}()
    scols = Vector{Int}()
    selements = Vector{ComplexF64}()

    P_sum_inner_dict = Dict(mi => sum(P[mi, mi] for P in values(P_dict)) for (mi, _) in sim.coupled_midxs)

    for (mi_μ, coupled_mi_v) in sim.coupled_midxs    # loop through all matrix indices (1st)
        mi_v = mi_μ     # only diagonal elements
        # now we start building the specific μv matrix element
        matrix_element = 0.0 + 0.0im
        # INSTEAD OF LOOPING OVER ALL INDICES, JUST TAKE THE ONES CONNECTED TO mi_v == mi_μ
        for mi_μ_p in coupled_mi_v
            connections_μp_v = sim.coupled_midxs_tunneling[SVector{2}(mi_μ_p, mi_v)]
            P_sum_inner = P_sum_inner_dict[mi_μ_p]

            for (η, V_μpvη, _) in connections_μp_v    # loop through all connected moire cells
                if isapprox(norm(η), 0, atol=1e-8) && mi_μ_p == mi_v    # deltas of the hamiltonian
                    continue
                end

                matrix_element += V_μpvη * P_sum_inner
            end
        end

        N_tot = sim.moire_N[1] * sim.moire_N[2]
        matrix_element /= N_tot

        if isapprox(matrix_element, 0, atol=1e-8)
            continue
        end

        push!(srows, mi_μ)
        push!(scols, mi_v)
        push!(selements, matrix_element)

    end

    Hartree_k = sparse(srows, scols, selements, dim, dim)
    return Hartree_k
end


function construct_Fock_k(
    k::SVector{3, Float64}, 
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    dim = 8 * (sim.geometry.mn[1]^2 + sim.geometry.mn[1] * sim.geometry.mn[2] + sim.geometry.mn[2]^2)    # needed for fixing sparse matrix dimensions
    srows = Vector{Int}()
    scols = Vector{Int}()
    selements = Vector{ComplexF64}()
    for (mi_μ, coupled_mi_v) in sim.coupled_midxs    # loop through all matrix indices (1st)

        for mi_v in coupled_mi_v    # loop through all connected matrix indices (2nd)
            connections_μv = sim.coupled_midxs_tunneling[SVector{2}(mi_μ, mi_v)]
            matrix_element = 0.0 + 0.0im

            # delta functions of the Fock term
            μ_parts = mi_to_parts(mi_μ, sim.geometry)
            v_parts = mi_to_parts(mi_v, sim.geometry)
            if μ_parts[4] != v_parts[4]    # opposite spins
                continue
            end
            if (μ_parts[1], μ_parts[2], μ_parts[3]) == (v_parts[1], v_parts[2], v_parts[3])
                continue
            end

            for (η, V_μpvη, _) in connections_μv    # loop through all connected moire cells
                
                P_sum_inner = 0.
                for (kp_idx, P_kp) in P_dict
                    kp = frac_to_cart(kp_idx[1] / sim.moire_N[1], kp_idx[2] / sim.moire_N[2], sim.geometry) 
                    P_sum_inner += P_kp[mi_v, mi_μ] * exp(-im * dot(kp - k, η))
                end

                matrix_element += V_μpvη * P_sum_inner
            end

            N_tot = sim.moire_N[1] * sim.moire_N[2]
            matrix_element /= -N_tot

            if isapprox(matrix_element, 0, atol=1e-8)
                continue
            end

            push!(srows, mi_μ)
            push!(scols, mi_v)
            push!(selements, matrix_element)

        end
    end

    Fock_k = sparse(srows, scols, selements, dim, dim)
    return Fock_k
end


"""Construct Hartree_k sparse matrix given k int coords and density matrix dictionary P_dict (keys are k integer coordinates)"""
function construct_Fock_k(
    k_idx::SVector{2, Int}, 
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    k = frac_to_cart(k_idx[1] / sim.moire_N[1], k_idx[2] / sim.moire_N[2], sim.geometry)
    return construct_Fock_k(k, P_dict, sim)
end


"""Construct the full HF hamiltonian k-block given tight binding term and density matrix blocks (k vector floats)"""
function get_HF_hamiltonian_k(
    k::SVector{3, Float64},
    T_k::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}},
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    H_H_k = construct_Hartree_k(P_dict, sim)
    H_F_k = construct_Fock_k(k, P_dict, sim)
    return T_k + H_H_k + H_F_k
end


"""Construct the full HF hamiltonian k-block given tight binding term and density matrix blocks (k integer index)"""
function get_HF_hamiltonian_k(
    k_idx::SVector{2, Int},
    T_k::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}},
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    k = frac_to_cart(k_idx[1] / sim.moire_N[1], k_idx[2] / sim.moire_N[2], sim.geometry)
    H_H_k = construct_Hartree_k(P_dict, sim)
    H_F_k = construct_Fock_k(k, P_dict, sim)
    return T_k + H_H_k + H_F_k
end



end