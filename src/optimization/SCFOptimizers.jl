module SCFOptimizers
export SCFOutput, SCF_optimize, plot_SCF_history
using LinearAlgebra, StaticArrays, SparseArrays, Random, Distributed, ChunkSplitters, FeastKit, BenchmarkTools, Random
import PyPlot as plt
using ..GeoMapping, ..Ansatze, ..Useful, ..Hamiltonians


struct SCFOutput
    H_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
    E_dict::Dict{SVector{2, Int}, Vector{Float64}}
    E_history::Vector{Float64}
    ϵ_error_history::Vector{Float64}
end


"""numerically stable Fermi function (avoids exp overflow (exp gets too big))"""
function FD_dist(E::Float64, μ::Float64, kT::Float64)::Float64
    if kT == 0.0
        return E > μ ? 0.0 : 1.0  
    end
    x = -(1 / kT) * (μ - E)
    return x > 0 ? exp(-x) / (1 + exp(-x)) : 1 / (1 + exp(x))
end


function find_mu(ε::Vector{Float64}, ne, N, β; tol=1e-12)
    F(μ) = sum(fermi.(ε .- μ, β)) / N - ne
    lo, hi = minimum(ε) - 10/β, maximum(ε) + 10/β
    while hi - lo > tol
        mid = (lo + hi) / 2
        F(mid) > 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end


"""find μ using a bisection method (n_avg is monotonic in μ so this method always converges)"""
function find_chemical_potential(E::Vector{Float64}, N_e::Int, kT::Float64; ϵtol=1e-12)
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


"""Get the converged HF hamiltonian (filling is inferred from trace(P_0))"""
function SCF_optimize(
    P_ansatz_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    T_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation;
    ϵ_tol::Float64 = 1e-8,
    maxiter::Int = 1000,
    kT::Float64 = 0.0,
    parallelized::Bool = false,     # keep to false when running on a cluster
    oda::Bool = false,
    mixing::Float64 = 1.0           # overriden if oda is set to true
)::SCFOutput

    if mixing < 0.0 || mixing > 1.0
        error("mixing value out of bounds, set it to be within [0., 1.]")
    end

    total_iters = 0
    E_history = Float64[]
    ϵ_error_history = Float64[]
    P_dict_current_iter = P_ansatz_dict
    N_e = round(Int, real(sum(tr.(values(P_dict_current_iter)))))
    converged = false

    P_dict_next_iter, HF_dict_next_iter = typeof(P_ansatz_dict), typeof(P_ansatz_dict)
    while total_iters < maxiter + 1
        total_iters += 1
        println("SCF iteration: $total_iters")

        @time HF_dict_current_iter = hamiltonian_term_dict(kidx -> get_HF_hamiltonian_k(kidx, T_dict[kidx], P_dict_current_iter, sim), sim)
        println("Calculated HF_dict_current_iter")

        ######################################

        eigenspectrum = Dict{SVector{2,Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()
        maxiter_loop = length(HF_dict_current_iter)
        iter = 0
        for (kidx, HF_k) in HF_dict_current_iter
            iter += 1
            F = eigen(Hermitian(Matrix(HF_k)))
            eigenspectrum[kidx] = (F.values, F.vectors)
            progress("Diagonalizing HF_k", iter, maxiter_loop)
        end

        # determine the occupations and construct new P
        allE = reduce(vcat, first.(values(eigenspectrum)))
        μ = find_chemical_potential(allE, N_e, kT)

        P_dict_next_iter = Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}()
        tot_iters = 0
        n_iters = length(eigenspectrum)
        for (kidx, (vals, vecs)) in eigenspectrum
            P_dict_next_iter[kidx] = vecs * Diagonal(FD_dist.(vals, μ, kT)) * vecs'
            tot_iters += 1
            progress("P_dict_next_iter constructed", tot_iters, n_iters)
        end

        #####################################

        @time HF_dict_next_iter = hamiltonian_term_dict(kidx -> get_HF_hamiltonian_k(kidx, T_dict[kidx], P_dict_next_iter, sim), sim)
        println("Calculated HF_dict_next_iter")
        E_next = HF_energy(P_dict_next_iter, HF_dict_next_iter)
        push!(E_history, E_next)
        ϵ_error = SCF_loss(P_dict_current_iter, P_dict_next_iter)
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
            P_k_mixed = P_dict_current_iter[kidx] * (1 - mixing) + P_dict_next_iter[kidx] * mixing
            P_dict_mixed[kidx] = P_k_mixed 
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
    for (kidx, _) in P_dict_next_iter
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
    mixing::Float64 = 1.0
)::SCFOutput
    P_dict = P_ansatz(filling, sim, type=type, rng=rng)
    T_dict = hamiltonian_term_dict(kidx -> construct_T_k(kidx, sim), sim)
    return SCF_optimize(P_dict, T_dict, sim, ϵ_tol=ϵ_tol, maxiter=maxiter, kT=kT, parallelized=parallelized, oda=oda, mixing=mixing)
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