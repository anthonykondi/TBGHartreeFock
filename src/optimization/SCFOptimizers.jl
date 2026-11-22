module SCFOptimizers
export SCFOutput, SCF_optimize, plot_SCF_history, save_scf_output, load_scf_output
using LinearAlgebra, StaticArrays, SparseArrays, Random, Distributed, ChunkSplitters, FeastKit, BenchmarkTools, Random, JLD2
import PyPlot as plt
using ..GeoMapping, ..Ansatze, ..Useful, ..Hamiltonians


struct SCFOutput
    H_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    E_dict::Dict{SVector{2, Int}, Vector{Float64}}
    E_history::Vector{Float64}
    ϵ_error_history::Vector{Float64}
end


"""
Save an SCFOutput to disk (JLD2). `metadata` is for anything you need to reproduce or
identify the run (θ, (m,n), HF_type, kT, ϵ_tol, cutoff, V parameters, ...).
"""
function save_scf_output(
    path::String,
    out::SCFOutput;
)::Nothing
    jldsave(path; compress = true,
        H_dict = out.H_dict,
        P_dict = out.P_dict,
        E_dict = out.E_dict,
        E_history = out.E_history,
        ϵ_error_history = out.ϵ_error_history,
    )
    return nothing
end


"""Load an SCFOutput saved with save_scf_output. Returns (output, metadata)."""
function load_scf_output(
    path::String
)::SCFOutput
    jldopen(path, "r") do f
        out = SCFOutput(
            Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}(f["H_dict"]),
            Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}(f["P_dict"]),
            Dict{SVector{2, Int}, Vector{Float64}}(f["E_dict"]),
            Vector{Float64}(f["E_history"]),
            Vector{Float64}(f["ϵ_error_history"]),
        )
        return out
    end
end


"""numerically stable Fermi function (avoids exp overflow (exp gets too big))"""
function FD_dist(E::Float64, μ::Float64, kT::Float64)::Float64
    if kT == 0.0
        return E > μ ? 0.0 : 1.0  
    end
    x = -(1 / kT) * (μ - E)
    return x > 0 ? exp(-x) / (1 + exp(-x)) : 1 / (1 + exp(x))
end


"""find μ using a bisection method (n_avg is monotonic in μ so this method always converges)"""
function find_chemical_potential(E::Vector{Float64}, N_e::Int, kT::Float64; ϵtol=1e-12)::Float64
    if kT == 0.0
        Es = sort(E); n = N_e
        n <= 0 && return Es[1] - 1.0
        n >= length(Es) && return Es[end] + 1.0
        return (Es[n] + Es[n+1]) / 2
    end
    lo, hi = minimum(E) - 50kT, maximum(E) + 50kT
    while hi - lo > ϵtol
        mid = (lo + hi) / 2
        sum(FD_dist.(E, mid, kT)) - N_e > 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end


function SCF_loss(
    P_dict_old::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    P_dict_new::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
)::Float64
    loss = 0.0
    for (kidx, _) in P_dict_old
        loss += norm(P_dict_old[kidx] - P_dict_new[kidx])
    end
    return loss
end


function SCF_oda(
    P_dict_old::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}, 
    P_dict_new::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    HF_dict_current_iter::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    HF_dict_next_iter::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
)::Float64
    s = 0.0; c = 0.0
    for (kidx, _) in P_dict_old
        s += real(2 * tr(HF_dict_current_iter[kidx] * (P_dict_new[kidx] - P_dict_old[kidx])))
        c += real(tr((HF_dict_next_iter[kidx] - HF_dict_current_iter[kidx]) * (P_dict_new[kidx] - P_dict_old[kidx])))
    end
    
    if 2 * c <= -s
        mixing = 1.0
    else
        mixing = clamp(-s/(2c), 0.0, 1.0)
    end
    
    return mixing
end


"""Impose the spin structure of HF_type on a starting density matrix (only used for P_0)."""
function enforce_spin_structure(
    P::Matrix{ComplexF64}, 
    up::Vector{Int}, 
    dn::Vector{Int},
    HF_type::String
)::Matrix{ComplexF64}
    HF_type == "GHF" && return P
    Q = zeros(ComplexF64, size(P))
    # these are valid 1rdm that preserve P^2=P, electron number, P'=P
    if HF_type == "UHF"
        Q[up, up] = P[up, up]
        Q[dn, dn] = P[dn, dn]
    else    # RHF
        Pavg = (P[up, up] + P[dn, dn]) / 2
        Q[up, up] = Pavg
        Q[dn, dn] = Pavg
    end
    return Q
end


"""Get the converged HF hamiltonian (filling is inferred from trace(P_0)). HF_type: "GHF", "UHF" or "RHF" """
function SCF_optimize(
    P_ansatz_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    T_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation;
    ϵ_tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    kT::Float64 = 0.0,
    parallelized::Bool = false,     # keep to false when running on a cluster
    oda::Bool = false,
    mixing::Float64 = 1.0,          # overriden if oda is set to true
    HF_type::String = "GHF"         # GHF, UHF, RHF
)::SCFOutput

    if mixing < 0.0 || mixing > 1.0
        error("mixing value out of bounds, set it to be within [0., 1.]")
    end
    HF_type in ("GHF", "UHF", "RHF") || error("HF_type must be \"GHF\", \"UHF\" or \"RHF\"")

    up, dn = HF_type == "GHF" ? (Int[], Int[]) : spin_partition(sim)

    # impose the spin structure on the starting guess
    P_dict_current_iter = typeof(P_ansatz_dict)()
    for (kidx, P_k) in P_ansatz_dict
        P_dict_current_iter[kidx] = enforce_spin_structure(Matrix(P_k), up, dn, HF_type)
    end

    N_e = round(Int, real(sum(tr.(values(P_dict_current_iter)))))
    if HF_type == "RHF" && isodd(N_e)
        error("RHF needs an even electron number, got N_e = $N_e")    # this can be easily ensured by having |moire_N| be a multiple of 4  
    end

    total_iters = 0
    E_history = Float64[]
    ϵ_error_history = Float64[]
    converged = false

    P_dict_next_iter, HF_dict_next_iter = typeof(P_ansatz_dict), typeof(P_ansatz_dict)
    while total_iters < maxiter + 1
        total_iters += 1
        println("SCF iteration: $total_iters")

        @time HF_dict_current_iter = hamiltonian_term_dict(kidx -> get_HF_hamiltonian_k(kidx, T_dict[kidx], P_dict_current_iter, sim), sim)
        println("Calculated HF_dict_current_iter")

        ######################################
        # diagonalize: full block (GHF) or per-spin blocks (UHF, RHF)
        eig_main = Dict{SVector{2,Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()  # GHF: full, UHF/RHF: up block
        eig_dn   = Dict{SVector{2,Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()  # UHF only
        maxiter_loop = length(HF_dict_current_iter)
        iter = 0
        for (kidx, HF_k) in HF_dict_current_iter   # ===================== [ADD MULTIPROCESSING HERE] =====================
            iter += 1
            H = Matrix(HF_k)
            if HF_type == "GHF"
                F = eigen(Hermitian(H))
                eig_main[kidx] = (F.values, F.vectors)
            else
                Fu = eigen(Hermitian(H[up, up]))
                eig_main[kidx] = (Fu.values, Fu.vectors)
                if HF_type == "UHF"
                    Fd = eigen(Hermitian(H[dn, dn]))
                    eig_dn[kidx] = (Fd.values, Fd.vectors)
                end
            end
            progress("Diagonalizing HF_k", iter, maxiter_loop)
        end

        # chemical potential from the levels that are actually allowed to be filled
        allE = reduce(vcat, first.(values(eig_main)))
        if HF_type == "UHF"
            allE = vcat(allE, reduce(vcat, first.(values(eig_dn))))
        end
        N_fill = HF_type == "RHF" ? N_e ÷ 2 : N_e       # RHF: only the up spectrum, half the electrons
        μ = find_chemical_potential(allE, N_fill, kT)

        # construct new P with the right block structure
        P_dict_next_iter = Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}()
        tot_iters = 0
        n_iters = length(eig_main)
        for (kidx, (vals, vecs)) in eig_main
            if HF_type == "GHF"
                P_dict_next_iter[kidx] = vecs * Diagonal(FD_dist.(vals, μ, kT)) * vecs'
            else
                P_k = zeros(ComplexF64, sim.geometry.N_tot, sim.geometry.N_tot)
                P_up = vecs * Diagonal(FD_dist.(vals, μ, kT)) * vecs'
                P_k[up, up] = P_up
                if HF_type == "RHF"
                    P_k[dn, dn] = P_up          # same occupied subspace for both spins
                else
                    vals_d, vecs_d = eig_dn[kidx]
                    P_k[dn, dn] = vecs_d * Diagonal(FD_dist.(vals_d, μ, kT)) * vecs_d'
                end
                P_dict_next_iter[kidx] = P_k
            end
            tot_iters += 1
            progress("P_dict_next_iter constructed", tot_iters, n_iters)
        end

        #####################################

        @time HF_dict_next_iter = hamiltonian_term_dict(kidx -> get_HF_hamiltonian_k(kidx, T_dict[kidx], P_dict_next_iter, sim), sim)
        println("Calculated HF_dict_next_iter")
        E_next = HF_energy(P_dict_next_iter, HF_dict_next_iter, sim)
        println("HF energy = $E_next")
        push!(E_history, E_next)
        ϵ_error = SCF_loss(P_dict_current_iter, P_dict_next_iter)
        println("ϵ_error = $ϵ_error")
        push!(ϵ_error_history, ϵ_error)

        if ϵ_error < ϵ_tol
            converged = true
            break
        end

        if total_iters == maxiter   # to skip the oda/mixing
            break
        end

        if oda
            mixing = SCF_oda(P_dict_current_iter, P_dict_next_iter, HF_dict_current_iter, HF_dict_next_iter)
            println("Performed ODA")
        end
        P_dict_mixed = Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}()
        for (kidx, _) in P_dict_next_iter
            P_dict_mixed[kidx] = P_dict_current_iter[kidx] * (1 - mixing) + P_dict_next_iter[kidx] * mixing
        end
        println("Mixed the density matrices")

        P_dict_current_iter = P_dict_mixed    # prepare for next iteration
    end

    if !converged
        @warn("SCF did not converge")
    end

    # P_dict_next_iter, HF_dict_next_iter are the converged/final values
    E_dict = Dict{SVector{2, Int}, Vector{Float64}}()
    n_iters = length(P_dict_next_iter)
    iter_now = 0
    for (kidx, _) in P_dict_next_iter    # ===================== [ADD MULTIPROCESSING HERE] =====================
        iter_now += 1
        eigvals, _ = eigen(Hermitian(Matrix(HF_dict_next_iter[kidx])))
        E_dict[kidx] = eigvals
        progress("Diagonalizing HF_out", iter_now, n_iters)
    end

    return SCFOutput(HF_dict_next_iter, P_dict_next_iter, E_dict, E_history, ϵ_error_history)
end


"""
Get the converged HF hamiltonian (specify filling and P_0 ansatz type).
Builds T_dict and P_ansatze_dict in here (wrapper function) and then passes them on to the main function.
"""
function SCF_optimize(
    filling::Float64,
    sim::TBGSimulation;
    type::String = "random",
    rng::AbstractRNG = Random.default_rng(),
    ϵ_tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    kT::Float64 = 0.0,
    parallelized::Bool = false,
    oda::Bool = false,
    mixing::Float64 = 1.0,
    HF_type::String = "GHF"         # GHF, UHF, RHF
)::SCFOutput
    P_dict = P_ansatz(filling, sim, type=type, rng=rng)
    T_dict = hamiltonian_term_dict(kidx -> construct_T_k(kidx, sim), sim)
    return SCF_optimize(P_dict, T_dict, sim, ϵ_tol=ϵ_tol, maxiter=maxiter, kT=kT, parallelized=parallelized, oda=oda, mixing=mixing, HF_type=HF_type)
end


"""Plot the SCF energy and convergence error history."""
function plot_SCF_history(scfout::SCFOutput; tag::String = "")::Nothing
    iters = 1:length(scfout.E_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.plot(iters, scfout.E_history, marker="o", ms=3, lw=1)
    ax1.set_xlabel("SCF iteration")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("Energy")

    ax2.semilogy(1:length(scfout.ϵ_error_history), scfout.ϵ_error_history, marker="o", ms=3, lw=1)
    ax2.set_xlabel("SCF iteration")
    ax2.set_ylabel("|P_out - P_in|")
    ax2.set_title("Convergence error")

    fig.tight_layout()
    plt.savefig("scf_history_" * tag * ".png")
    plt.show()
    return nothing
end


end