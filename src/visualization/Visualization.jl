module Visualization 
export plot_thresh_sites, test_thresh_construction_scaling
using PyPlot, StaticArrays, BenchmarkTools
using ..KspGeoMapping, ..Structure


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



end