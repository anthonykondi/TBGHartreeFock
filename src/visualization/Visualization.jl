module Visualization 
export plot_thresh_sites, test_thresh_construction_scaling, visualize_1rdm_k_block, plot_magnetization, plot_density, local_density_matrix, site_observables
using PyPlot, StaticArrays, BenchmarkTools, SparseArrays, LinearAlgebra
using ..GeoMapping, ..Structure


function plot_thresh_sites(
    sim::TBGSimulation; 
    N_moire::SVector{2, Int} = SVector{2}(1, 1),
    tag::String = ""
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
    ax.set_title("θ = $(tbg_geom.θ * (180 / pi)), N_thresh_sites = $(size(connected_sites_pos)[1])\n" * tag)

    plt.savefig("thresh_sites_" * tag * ".png")
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
    ax.set_title("1-RDM k-block (darker: larger |P|, white = 0, max = $(round(max_mag, sigdigits=3)))\n" * tag)

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


"""
Average of the k-blocks: the on-site density matrix (identical in every moiré cell).
This is the fourier transform from k to real space, where R = R' (η = 0).
Use this specifically for calculating observables over a unit cell (only need the diagonal for the trace).
"""
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
Per-atom observables (one entry per (graphene cell, sublattice, layer)), for moiré cell R = 0.
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
    # these are local spins
    return (
        orbitals = orbitals,
        n  = n_up .+ n_dn,
        sz = (n_up .- n_dn) ./ 2,
        sx = real.(P_ud),
        sy = -imag.(P_ud),
    )
end


function plot_site_map(
    sim::TBGSimulation,
    orbitals::Vector,
    values::Vector{Float64};
    N_moire::SVector{2, Int} = SVector{2}(1, 1),
    tile_values::Bool = false,
    label::String = "value",
    signed::Bool = true,
    cmap::String = signed ? "bwr" : "viridis",
    vmax::Union{Nothing, Float64} = nothing,
    layers::Vector{Int} = [1, 2],
    show_cell::Bool = true,
    savepath::String = ""
)::Nothing
    geom = sim.geometry

    # normalise to 1-based (graphene, sublattice, layer)
    o1 = minimum(o[1] for o in orbitals)
    o2 = minimum(o[2] for o in orbitals)
    o3 = minimum(o[3] for o in orbitals)
    orb = [(g - o1 + 1, s - o2 + 1, l - o3 + 1) for (g, s, l) in orbitals]

    vm = isnothing(vmax) ? maximum(abs.(values)) : vmax
    vm = vm > 0 ? vm : 1.0
    vmin = signed ? -vm : 0.0

    # moiré lattice vectors from the displacement of one atom under R -> R + (1,0), (0,1)
    g0 = geom.graphene_cells[1][1]
    p0 = SiteIdx(SVector(0, 0), g0, 1, 1, 1, geom).pos
    L1 = (SiteIdx(SVector(1, 0), g0, 1, 1, 1, geom).pos - p0)[1:2]
    L2 = (SiteIdx(SVector(0, 1), g0, 1, 1, 1, geom).pos - p0)[1:2]
    Linv = inv([L1 L2])                      # columns are L1, L2

    # position in the home cell, wrapped into the parallelogram spanned by L1, L2
    function wrapped_pos(g, s, l)
        p = SiteIdx(SVector(0, 0), geom.graphene_cells[l][g], s, l, 1, geom).pos[1:2]
        f = Linv * p
        f = f .- floor.(f .+ 1e-9)           # fractional coords in [0, 1)
        return [L1 L2] * f
    end

    layer_color  = Dict(1 => "tab:blue", 2 => "tab:red")
    layer_marker = Dict(1 => "o", 2 => "D")
    layer_size   = Dict(1 => 170, 2 => 70)
    cell_range = [(mx, my) for mx in -N_moire[1]:N_moire[1] for my in -N_moire[2]:N_moire[2]]

    fig, ax = subplots(figsize=(8, 7))

    # ---------- underlay: wrapped cell tiled by moiré vectors ----------
    for l in layers
        xs = Float64[]; ys = Float64[]
        for g in eachindex(geom.graphene_cells[l]), s in 1:2
            base = wrapped_pos(g, s, l)
            for (mx, my) in cell_range
                q = base .+ mx .* L1 .+ my .* L2
                push!(xs, q[1]); push!(ys, q[2])
            end
        end
        ax.scatter(xs, ys, s=12, marker=".", color=layer_color[l], alpha=0.4,
                   zorder=1, label="layer $l lattice")
    end

    # ---------- moiré cell outline ----------
    if show_cell
        ax.plot([0, L1[1], L1[1] + L2[1], L2[1], 0],
                [0, L1[2], L1[2] + L2[2], L2[2], 0], "--", color="k", lw=1.0, zorder=2)
    end

    # ---------- overlay: values ----------
    sc = nothing
    for (mx, my) in (tile_values ? cell_range : [(0, 0)])
        home = (mx == 0 && my == 0)
        for l in sort(layers)
            idx = findall(o -> o[3] == l, orb)
            px = Float64[]; py = Float64[]
            for i in idx
                g, s, _ = orb[i]
                q = wrapped_pos(g, s, l) .+ mx .* L1 .+ my .* L2
                push!(px, q[1]); push!(py, q[2])
            end
            sc = ax.scatter(px, py, c=values[idx], s=layer_size[l], marker=layer_marker[l],
                            cmap=cmap, vmin=vmin, vmax=vm,
                            alpha=home ? 1.0 : 0.3,
                            edgecolors=home ? "k" : "none", linewidths=0.7, zorder=3 + l)
        end
    end

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)
    ax.scatter([], [], s=100, marker="o", facecolors="none", edgecolors="k", label="layer 1 (circles)")
    ax.scatter([], [], s=50,  marker="D", facecolors="none", edgecolors="k", label="layer 2 (diamonds)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title("θ = $(round(geom.θ * 180 / π, digits=2)), $label")
    tight_layout()
    !isempty(savepath) && plt.savefig(savepath, dpi=200)
    plt.show()
    return nothing
end


function plot_magnetization(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation;
    component::Symbol = :z,
    kwargs...
)::Nothing
    component in (:x, :y, :z, :norm) || error("component must be :x, :y, :z or :norm")
    obs = site_observables(P_dict, sim)
    m = component == :z ? obs.sz :
        component == :x ? obs.sx :
        component == :y ? obs.sy :
        sqrt.(obs.sx .^ 2 .+ obs.sy .^ 2 .+ obs.sz .^ 2)
    label = component == :norm ? "|S|" : "S_$(component)"
    plot_site_map(sim, collect(obs.orbitals), Vector{Float64}(m);
                  label = label, signed = component != :norm, kwargs...)
    return nothing
end


function plot_density(
    P_dict::Dict{SVector{2, Int}, Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}},
    sim::TBGSimulation;
    relative::Bool = false,
    kwargs...
)::Nothing
    obs = site_observables(P_dict, sim)
    vals = relative ? obs.n .- 1.0 : obs.n
    plot_site_map(sim, collect(obs.orbitals), Vector{Float64}(vals);
                  label = relative ? "n - 1" : "n", signed = relative, kwargs...)
    return nothing
end



end