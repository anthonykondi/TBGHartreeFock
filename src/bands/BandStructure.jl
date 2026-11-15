module BandStructure
export plot_TB_band_structure, plot_HF_band_structure, get_dos

using PyPlot, BenchmarkTools, StaticArrays, LinearAlgebra

using ..GeoMapping, ..Hamiltonians, ..SCFOptimizers, ..Useful


"""
Calculate density of states using a gaussian for the dirac delta factors.
Normalization is the number of bands in the system.
"""
function get_dos(Energies_dict::Dict{SVector{2, Int}, Vector{Float64}}; η::Float64 = 0.01)::Matrix{Float64}
    gaussian(x, η) = exp(-x^2 / (2η^2)) / (sqrt(2π) * η)

    Energies = reduce(vcat, values(Energies_dict))
    N_tot = length(Energies_dict)

    E_range = range(minimum(Energies) - 5η, maximum(Energies) + 5η, step = η / 4)
    dos_values = [sum(gaussian(E - ε, η) for ε in Energies) / N_tot for E in E_range]

    return [collect(E_range) dos_values]
end



"""
Piecewise-linear path through the high-symmetry points named in `path`,
e.g. [:K, :Γ, :M, :Kp]. Returns (ks, k_idxs, xs, ticks, labels).

k_idxs[i] is the (i1, i2) grid index of the k-point on the nk x nk
Monkhorst-Pack grid nearest to `ks[i]` (path points that don't land exactly
on the grid get snapped to the nearest grid point, meant for HF simulation).
"""
function build_bz_path(
    sim::TBGSimulation,
    path::Vector{Symbol};
    npts_per_unit::Int = 150,
)::Tuple{Vector{SVector{3, Float64}}, Vector{SVector{2, Int}}, Vector{Float64}, Vector{Float64}, Vector{String}}

    geom = sim.geometry
    lookup = Dict{Symbol, Tuple{SVector{3, Float64}, String}}(
        :K     => (geom.kK,  "K"),
        :Kp    => (geom.kKp, "K'"),
        :Gamma => (geom.kΓ,  "Γ"),
        :Γ     => (geom.kΓ,  "Γ"),
        :M     => (geom.kM,  "M"),
    )

    for s in path
        haskey(lookup, s) || error("Unknown symmetry point :$s. Options: $(sort(collect(keys(lookup))))")
    end

    symmetry_pts = [lookup[s][1] for s in path]
    labels       = [lookup[s][2] for s in path]
    scale        = norm(geom.k1)

    N1, N2 = sim.moire_N
    B = hcat(geom.k1[1:2], geom.k2[1:2])      # columns = reciprocal basis vectors

    # k = (i1/N1) k1 + (i2/N2) k2  ->  nearest (i1, i2), wrapped into 0..N-1
    function k_to_idx(k::SVector{3, Float64})::SVector{2, Int}
        frac = B \ k[1:2]                      # k = frac[1] k1 + frac[2] k2
        i1 = mod(round(Int, frac[1] * N1), N1)
        i2 = mod(round(Int, frac[2] * N2), N2)
        return SVector{2, Int}(i1, i2)
    end

    ks     = SVector{3, Float64}[]
    k_idxs = SVector{2, Int}[]
    xs     = Float64[]
    ticks  = [0.0]
    x0     = 0.0

    for i in 1:(length(symmetry_pts) - 1)
        ka, kb = symmetry_pts[i], symmetry_pts[i+1]
        L = norm(kb - ka)
        n = max(round(Int, npts_per_unit * L / scale), 2)

        for t in range(0, 1; length = n + 1)[1:end-1]
            k = ka + t * (kb - ka)
            push!(ks, k)
            push!(k_idxs, k_to_idx(k))
            push!(xs, x0 + t * L)
        end
        x0 += L
        push!(ticks, x0)
    end

    push!(ks, symmetry_pts[end])
    push!(k_idxs, k_to_idx(symmetry_pts[end]))
    push!(xs, x0)

    return ks, k_idxs, xs, ticks, labels
end


"""
Plot the (TB only) band structure through high symmetry points in 1st BZ.
"""
function plot_TB_band_structure(
    sim::TBGSimulation; 
    npts_per_unit::Int = 150,
    ylims = nothing, 
    bz_path::Vector{Symbol} = [:K, :Gamma, :M, :Kp],
    tag::String = "",
    parallelized::Bool = false      # keep to false when running on a cluster [TEST CHUNKSPLITTERS BY IMPLEMENTING IT HERE]
)::Nothing
    # first, get your k points in the BZ path
    ks, _, xs, ticks, labels = build_bz_path(sim, bz_path; npts_per_unit)

    # then, diagonalize and plot the bands
    mi_length = length(sim.coupled_midxs)
    Energies = zeros(mi_length, length(ks))   # each columns = E(k) where rows label bands
    
    t_iter = 0
    num_iters = length(ks) 
    for (i, k) in enumerate(ks)
        T_k = construct_T_k(k, sim)
        Energies[:, i] = eigvals(Hermitian(Matrix(T_k)))   # dump the eigenvalues on the entire column
        t_iter += 1
        progress("Diagonalizing T", t_iter, num_iters)
    end

    # plots each column as a separate plot (transpose to get the shape right)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, Energies', lw=0.5, alpha=1.0)          # Energies' -> shape (n_k, n_bands)
    
    # labels and other details
    for t in ticks
        ax.axvline(t, color="gray", lw=0.5)
    end
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(ticks[1], ticks[end])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("TB only, θ = $(round(sim.geometry.θ * (180 / pi), digits=2))°, [(m,n) = ($(sim.geometry.mn[1]), $(sim.geometry.mn[2]))]")
    ylims === nothing || ax.set_ylim(ylims...)    # if limits are specified use them
    fig.tight_layout()
    
    plt.savefig("bands_TB_only_mn=$(sim.geometry.mn[1])_$(sim.geometry.mn[2])_" * tag * ".png")
    plt.show()
end


"""
Plot the (full HF) band structure through high symmetry points in 1st BZ.
"""
function plot_HF_band_structure(
    Energies_dict::Dict{SVector{2, Int}, Vector{Float64}},
    sim::TBGSimulation; 
    npts_per_unit::Int = 150, 
    ylims = nothing, 
    bz_path::Vector{Symbol} = [:K, :Gamma, :M, :Kp],
    tag::String = ""
)::Nothing
    # first, get your k points in the BZ path
    _, k_idxs, xs, ticks, labels = build_bz_path(sim, bz_path; npts_per_unit)
    
    # take the energies in the BZ path (nearest ones, based on the build_bz_path function)
    mi_length = length(sim.coupled_midxs)
    Energies = zeros(mi_length, length(k_idxs))
    for (i, k_idx) in enumerate(k_idxs)
        Energies[:, i] = Energies_dict[k_idx]   # dump the eigenvalues on the entire column
    end

    # plots each column as a separate plot (transpose to get the shape right)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, Energies', lw=0.5, alpha=1.0)          # Energies' -> shape (n_k, n_bands)
    
    # labels and other details
    for t in ticks
        ax.axvline(t, color="gray", lw=0.5)
    end
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(ticks[1], ticks[end])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Full HF, θ = $(round(sim.geometry.θ * (180 / pi), digits=2))°, [(m,n) = ($(sim.geometry.mn[1]), $(sim.geometry.mn[2]))]")
    ylims === nothing || ax.set_ylim(ylims...)    # if limits are specified use them
    fig.tight_layout()
    
    plt.savefig("bands_full_HF_mn=$(sim.geometry.mn[1])_$(sim.geometry.mn[2])_" * tag * ".png")
    plt.show()
end


end