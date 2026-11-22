module Visualization 
export plot_thresh_sites, test_thresh_construction_scaling, visualize_1rdm_k_block, local_density_matrix, site_observables, plot_site_observable
using PyPlot, StaticArrays, BenchmarkTools, SparseArrays, LinearAlgebra
using ..GeoMapping, ..Structure


function plot_thresh_sites(
    sim::TBGSimulation; 
    N_moire::SVector{2, Int} = SVector{2}(1, 1)
)::Nothing
    tbg_geom = sim.geometry
    tbg_consts = sim.constants
    orig_site = SiteIdx(SVector(0, 0), SVector(0, 0), 1, 1, 1, tbg_geom)
    N1, N2 = N_moire

    fig, ax = plt.subplots()

    # =================================================================
    # plot the entire lattice over top as well

    for layer in 1:2
        lattice_sites = []
        for moire_x in -N1:N1, moire_y in -N2:N2
            moire = SVector{2}(moire_x, moire_y)
            for graphene in tbg_geom.graphene_cells[layer]
                for sublattice in 1:2
                    site_idx = SiteIdx(moire, graphene, sublattice, layer, 1, tbg_geom)
                    site_pos = site_idx.pos
                    push!(lattice_sites, site_pos)
                end
            end
        end
        # plot each layer separately (different colors)
        xs_2 = [v[1] for v in lattice_sites]
        ys_2 = [v[2] for v in lattice_sites]

        ax.scatter(xs_2, ys_2, marker=".", alpha=0.4)
    end

    # =================================================================

    # convert orig_site from SiteIdx to RelativeIdx object
    connected_sites = tbg_consts.thresh_connections[orig_site]
    
    connected_sites_pos = []
    for other_site in connected_sites
        push!(connected_sites_pos, other_site.pos)
    end

    xs = [v[1] for v in connected_sites_pos]
    ys = [v[2] for v in connected_sites_pos]

    ax.scatter(xs, ys, marker=".", alpha=0.5)

    ax.scatter(orig_site.pos[1], orig_site.pos[2], marker="o")

    ax.set_aspect("equal")
    ax.set_title("θ = $(tbg_geom.θ * (180 / pi)), N_thresh_sites = $(size(connected_sites_pos)[1])")

    plt.show()
end


function test_thresh_construction_scaling(sim::TBGSimulation)::Nothing
    tbg_geom = sim.geometry
    times = []
    m_interval = 1:32
    for i in m_interval
        println("Starting (m, n) = ($i, $(i+1))")
        tbg_geom, t1 = @timed TBGGeometry(i, i+1) 
        _, t2 = @timed TBGConstants(tbg_geom)
        t = t1 + t2
        push!(times, t)

        println("Ending (m, n) = ($i, $(i+1)), t = $t")
    end

    plt.plot(m_interval, times)
    plt.xlabel("m")
    plt.ylabel("time elapsed [s]")
    plt.savefig("time for thresh construction (1).png")
end


"""
Plot a k-block of the 1-RDM as a complex-valued color map.
Hue = phase of P[i,j] (arg ∈ [-π, π]); white = 0, darker/more saturated = larger |P[i,j]|.
Magnitude is normalized to max|P|. Rows are the first index, columns the second.
"""
function visualize_1rdm_k_block(
    P::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}};
    tag::String = "",
    darkness::Float64 = 0.6    # in [0, 1): how dark the largest entries get (0 = just saturated, no darkening)
)::Nothing
    M = Matrix(P)
    n_rows, n_cols = size(M)

    mag = abs.(M)
    max_mag = maximum(mag)
    phase = angle.(M)
    m = max_mag > 0 ? mag ./ max_mag : zeros(n_rows, n_cols)

    # HSV image: hue from phase, saturation = m (white at 0), value = 1 - darkness*m (darker at large m)
    hsv = Array{Float64}(undef, n_rows, n_cols, 3)
    hsv[:, :, 1] .= mod.((phase .+ π) ./ (2π), 1.0)
    hsv[:, :, 2] .= m
    hsv[:, :, 3] .= 1.0 .- darkness .* m
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    fig, ax = subplots(figsize=(7, 6))
    ax.imshow(rgb, interpolation="nearest", origin="upper")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    ax.set_title("1-RDM k-block (darker: larger |P|, white = 0, max = $(round(max_mag, sigdigits=3)))")

    # phase colorbar, using the same hue mapping as the image (matplotlib's "hsv" cmap)
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-π, vmax=π),
        cmap="hsv"
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[-π, -π/2, 0, π/2, π])
    cbar.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cbar.set_label("phase (shown at full saturation)")

    tight_layout()
    plt.savefig("1rdm_visual_" * tag * ".png")
    plt.show()
    return nothing
end


"""Average of the k-blocks: the on-site density matrix (identical in every moiré cell)."""
function local_density_matrix(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}}
)::Matrix{ComplexF64}
    dim = size(first(values(P_dict)), 1)
    P̄ = zeros(ComplexF64, dim, dim)
    for P_k in values(P_dict)
        P̄ .+= Matrix(P_k)
    end
    return P̄ ./ length(P_dict)
end


"""
Per-orbital observables (one entry per (graphene cell, sublattice, layer)), for moiré cell R = 0.
Returns (orbitals, n, sz, sx, sy), where orbitals[i] = (graphene, sublattice, layer).
"""
function site_observables(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation
)
    P̄ = local_density_matrix(P_dict)
    up, dn = spin_partition(sim)
    orbitals = [mi_to_parts(mi, sim.geometry)[1:3] for mi in up]
    n_up = real.(diag(P̄)[up])
    n_dn = real.(diag(P̄)[dn])
    P_ud = [P̄[up[i], dn[i]] for i in eachindex(up)]
    return (
        orbitals = orbitals,
        n  = n_up .+ n_dn,
        sz = (n_up .- n_dn) ./ 2,
        sx = real.(P_ud),
        sy = -imag.(P_ud),
    )
end


"""
Real-space map of a per-orbital quantity over the lattice.
The cell at moiré index (0,0) is drawn at full opacity, and the neighbouring cells (same values, by
periodicity) are faded. Signed quantities use a diverging colormap centred on 0, so brightness
(distance from white) is the magnitude. Marker size also scales with |value|.
"""
function plot_site_observable(
    sim::TBGSimulation,
    orbitals::Vector,
    values::Vector{Float64};
    N_moire::SVector{2, Int} = SVector{2}(0, 0),    # number of extra cells drawn on each side
    label::String = "value",
    signed::Bool = true,
    cmap::String = signed ? "bwr" : "viridis"
)::Nothing
    geom = sim.geometry
    vmax = maximum(abs.(values))
    vmax = vmax > 0 ? vmax : 1.0
    vmin = signed ? -vmax : 0.0
    sizes = 10 .+ 90 .* abs.(values) ./ vmax

    fig, ax = subplots(figsize=(7, 6))
    sc = nothing
    for mx in -N_moire[1]:N_moire[1], my in -N_moire[2]:N_moire[2]
        moire = SVector{2, Int}(mx, my)
        xs = Float64[]; ys = Float64[]
        for (g, s, l) in orbitals
            pos = SiteIdx(moire, g, s, l, 1, geom).pos
            push!(xs, pos[1]); push!(ys, pos[2])
        end
        is_home = (mx == 0 && my == 0)
        sc = ax.scatter(xs, ys, c=values, s=sizes, cmap=cmap, vmin=vmin, vmax=vmax,
                        alpha=is_home ? 1.0 : 0.3,
                        edgecolors=is_home ? "k" : "none", linewidths=0.4)
    end

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)
    ax.set_aspect("equal")
    ax.set_title("θ = $(round(geom.θ * 180 / π, digits=2)), $label")
    tight_layout()
    display(fig)
    return nothing
end



end