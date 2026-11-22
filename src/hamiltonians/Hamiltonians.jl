module Hamiltonians
export construct_T_k, construct_Hartree_k, construct_Fock_k, get_HF_hamiltonian_k, hamiltonian_term_dict, HF_energy

using LinearAlgebra, StaticArrays, SparseArrays 

using ..Useful, ..GeoMapping


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
Note that the Hartree mean field hamiltonian does not depend on k-point (k in call signature for compatibility with the dictionary constructor function)
"""
function construct_Hartree_k(
    k::SVector{3, Float64},
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


function construct_Hartree_k(
    k_idx::SVector{2, Int}, 
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    sim::TBGSimulation
)::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}
    k = frac_to_cart(k_idx[1] / sim.moire_N[1], k_idx[2] / sim.moire_N[2], sim.geometry)
    return construct_Hartree_k(k, P_dict, sim)
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
            # if (μ_parts[1], μ_parts[2], μ_parts[3]) == (v_parts[1], v_parts[2], v_parts[3])
            #     continue
            # end

            for (η, V_μpvη, _) in connections_μv    # loop through all connected moire cells
                if isapprox(norm(η), 0, atol=1e-8) && mi_μ == mi_v    # deltas of the hamiltonian (Claude pointed this out)
                    continue
                end

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
    H_H_k = construct_Hartree_k(k, P_dict, sim)
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
    H_H_k = construct_Hartree_k(k, P_dict, sim)
    H_F_k = construct_Fock_k(k, P_dict, sim)
    return T_k + H_H_k + H_F_k
end


"""
Get the k_idx => H dictionary for some hamiltonian term or the entire hamiltonian (use anonymous functions for setting the constructor)
Usage ex: HF_dict = Hamiltonian_term_dictz(kidx -> get_HF_hamiltonian_k(kidx, T_k, P_dict, sim), sim), 
where here kidx -> get_HF_hamiltonian_k(kidx, T_k, P_dict, sim) is the term_constructor.
"""
function hamiltonian_term_dict(
    term_constructor::Base.Callable, 
    sim::TBGSimulation
)::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    H_dict = Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}()
    terms_calculated = 0
    n_terms = sim.moire_N[1] * sim.moire_N[2]
    for kidx1 in 0:(sim.moire_N[1] - 1), kidx2 in 0:(sim.moire_N[2] - 1)    # ===================== [ADD MULTIPROCESSING HERE] =====================
        kidx = SVector(kidx1, kidx2)
        H_dict[kidx] = term_constructor(kidx)
        terms_calculated += 1
        progress("hamiltonian_term_dict terms calculated", terms_calculated, n_terms)
    end
    return H_dict
end


"""
Double-counting constant E_dc = -<H_int>_0 (total, not per moiré cell).

E_dc = -(N/2) Σ_{μ,v,η} V^{μν}_η (1 - δ_{η0}δ_{μν}) [ n_μ n_v - δ_{σ_μσ_v} |G_{νμ}(η)|² ]

n_μ      = (1/N) Σ_k P_k[μ,μ]
G_{νμ}(η) = (1/N) Σ_k P_k[v,μ] e^{-i k·η}   (the same inner sum as in construct_Fock_k)
"""
function HF_constant_shift(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation
)::Float64
    N_tot = sim.moire_N[1] * sim.moire_N[2]

    # k-points in cartesian coordinates, computed once
    ks = [(frac_to_cart(kidx[1] / sim.moire_N[1], kidx[2] / sim.moire_N[2], sim.geometry), P_k)
          for (kidx, P_k) in P_dict]

    # orbital densities per moiré cell
    n = Dict(mi => real(sum(P[mi, mi] for P in values(P_dict))) / N_tot
             for (mi, _) in sim.coupled_midxs)

    E_hartree = 0.0
    E_fock = 0.0

    for (mi_μ, coupled_mi_v) in sim.coupled_midxs
        μ_parts = mi_to_parts(mi_μ, sim.geometry)

        for mi_v in coupled_mi_v
            connections = sim.coupled_midxs_tunneling[SVector{2}(mi_μ, mi_v)]
            v_parts = mi_to_parts(mi_v, sim.geometry)
            same_spin = μ_parts[4] == v_parts[4]

            for (η, V_μvη, _) in connections
                # i ≠ j exclusion (same orbital in the same moiré cell)
                if mi_μ == mi_v && isapprox(norm(η), 0, atol=1e-8)
                    continue
                end

                # Hartree piece: V n_μ n_ν
                E_hartree += real(V_μvη) * n[mi_μ] * n[mi_v]

                # Exchange piece: V |G_{νμ}(η)|², same spin only
                if same_spin
                    G = 0.0 + 0.0im
                    for (k, P_k) in ks
                        G += P_k[mi_v, mi_μ] * exp(-im * dot(k, η))
                    end
                    G /= N_tot
                    E_fock += real(V_μvη) * abs2(G)
                end
            end
        end
    end

    return -(N_tot / 2) * (E_hartree - E_fock)
end


"""Get the non-interacting system energy from the 1rdm and the hamiltonian (works for any 1-body hamiltonian)"""
function HF_energy(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    HF_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation
)::Float64
    Energy = 0.0
    terms_calculated = 0
    n_terms = length(P_dict)
    for (kidx, P_k) in P_dict                   
        hamiltonian_k = HF_dict[kidx]
        Energy += real(tr(hamiltonian_k * P_k))
        terms_calculated += 1
        progress("HF_energy terms calculated", terms_calculated, n_terms)
    end
    return Energy + HF_constant_shift(P_dict, sim)
end



end