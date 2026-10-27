using StaticArrays
using LinearAlgebra
import PyPlot as plt
using BenchmarkTools
using ChunkSplitters
using Distributed
using SparseArrays


# note that the graphene indices swap roles on the second layer (should only need to worry about this in geometry construction [just amounts to a technical detail])

"""
Note that there are two indexing types:
    - Matrix indices: integers, used directly to get elements of matrices
    - Site indices: abstract structs, meant to serve as intermediates for matrix indices (~~"virtual indices"~~)
Matrix indices only have within moire cell info, unlike site indices which have all the info.
This is not a problem as the code is streamlined this way by letting the site indices do all the work
in the backround at the start to create the necessary coupling dictionaries, and then the matrix indices
do the rest of the work (much faster and less garbage collection, with the cost of more memory allocation).
"""

# ===============================================================================
# Geometry
# ===============================================================================


"""Geometry and stored in a common place"""
struct TBGGeometry
    d::Float64
    h::Float64 
    mn::SVector{2, Int}
    graphene_cells::Dict{Int, Vector{SVector{2, Int}}}
    θ::Float64
    a1_down::SVector{3, Float64}
    a1_up::SVector{3, Float64} 
    a2_down::SVector{3, Float64} 
    a2_up::SVector{3, Float64} 
    b1_down::SVector{3, Float64} 
    b1_up::SVector{3, Float64} 
    b2_down::SVector{3, Float64} 
    b2_up::SVector{3, Float64} 
    L1::SVector{3, Float64}
    L2::SVector{3, Float64}
    k1::SVector{3, Float64}
    k2::SVector{3, Float64}
end


# Using this as an intermediate in generating TBGConstants or for conveniently plotting lattice 
"""Label the sites in tbg with 5 dimentional multi-indices (position calculated by constructor)"""
struct SiteIdx         
    moire::SVector{2, Int}
    graphene::SVector{2, Int}
    sublattice::Int               # 1 or 2 (for b1 or b2 vectors)
    layer::Int                    # 1 or 2
    spin::Int                     # +1 or -1
    pos::SVector{3, Float64}      
end


"""Other Constants stored in a common place"""
struct TBGConstants
    t0::Float64 
    t1::Float64 
    q_sigma::Float64 
    q_pi::Float64
    r_c::Float64 
    l_c::Float64
    r_thresh::Float64 
    n_thresh::Int                 
    thresh_connections::Dict{SiteIdx, Vector{SiteIdx}}    # only includes spin up (need to account for this later)
end


"""Get the site indices of graphene cells in a moire cell, for layers 1 and 2"""
function get_graphene_cells(mn::SVector{2, Int})::Dict{Int, Vector{SVector{2,Int}}}
    m, n = mn
    R = n + m 
    N = m^2 + m*n + n^2
    graphene_in_moire = Dict{Int, Vector{SVector{2,Int}}}()

    # layer 1
    layer_1_graphene = []
    for i in -R:R, j in 0:2*R
        s1 = (n + m) * i + m * j
        s2 = -m * i + n * j
        if 0 <= s1 < N && 0 <= s2 < N
            push!(layer_1_graphene, [i, j])
        end
    end
    graphene_in_moire[1] = layer_1_graphene

    # layer 2
    layer_2_graphene = []
    for i in -R:R, j in 0:2*R
        s1 = (m + n) * i + n * j
        s2 = -n * i + m * j
        if 0 <= s1 < N && 0 <= s2 < N
            push!(layer_2_graphene, [i, j])
        end
    end
    graphene_in_moire[2] = layer_2_graphene

    return graphene_in_moire
end


"""Get the twist angle based on the commensurate lattice numbers (m,n)"""
function get_twist_angle(mn::SVector{2, Int})::Float64
    m, n = mn
    θ = acos((n^2 + 4*n*m + m^2) / (2 * (n^2 + n*m + m^2)))
    return θ
end


function rot_matrix_xy(θ::Float64)::Matrix{Float64}
    R_θ = [cos(θ) -sin(θ) 0;
           sin(θ) cos(θ) 0;
           0 0 1]
    return R_θ
end


"""constructor for the TBGGeometry object"""
function TBGGeometry(m::Int, n::Int)::TBGGeometry
    0 < m < n || throw(ArgumentError("require 0 < m < n"))

    d = 1.42
    h = 3.34
    mn = SVector{2}(m, n)
    graphene_cells = get_graphene_cells(mn)
    θ = get_twist_angle(mn)

    a1_down = SVector{3}((d*sqrt(3)) * [1, 0, 0])
    a1_up = SVector{3}(rot_matrix_xy(-θ) * a1_down)
    a2_down = SVector{3}((d*sqrt(3)) * [1/2, sqrt(3)/2, 0])
    a2_up = SVector{3}(rot_matrix_xy(-θ) * a2_down)

    b1_down = SVector{3}(d * [sqrt(3)/2, 1/2, 0])
    b1_up = SVector{3}(rot_matrix_xy(-θ) * b1_down)
    b2_down = SVector{3}(d * [sqrt(3)/2, -1/2, 0])
    b2_up = SVector{3}(rot_matrix_xy(-θ) * b2_down)

    L1 = SVector{3}(n * a1_down + m * a2_down)
    L2 = SVector{3}(-m * a1_down + (n + m) * a2_down)

    # reciprocal lattice vectors
    A = 2 * pi / (L1[1] * L2[2] - L1[2] * L2[1])
    k1 = SVector{3}(A * [L2[2], -L2[1], 0])
    k2 = SVector{3}(A * [-L1[2], L1[1], 0])

    return TBGGeometry(d, h, mn, graphene_cells, θ, a1_down,a1_up, a2_down, a2_up, b1_down, b1_up, b2_down, 
                        b2_up, L1, L2, k1, k2)            
end


"""Converts from site indices to real space position vector using tbg geometry"""
function get_position(moire::SVector{2, Int}, graphene::SVector{2, Int}, sublattice::Int,
                      layer::Int, tbg_geom::TBGGeometry)::SVector{3, Float64}
    h = tbg_geom.h
    L1 = tbg_geom.L1
    L2 = tbg_geom.L2

    if layer == 1
        a1 = tbg_geom.a1_down
        a2 = tbg_geom.a2_down
        b1 = tbg_geom.b1_down
        b2 = tbg_geom.b2_down
    else
        a1 = tbg_geom.a1_up
        a2 = tbg_geom.a2_up
        b1 = tbg_geom.b1_up
        b2 = tbg_geom.b2_up
    end

    # building the position vector
    position_vector = moire[1] * L1 + moire[2] * L2 
    position_vector += graphene[1] * a1 + graphene[2] * a2
    if sublattice == 1
        position_vector += b1
    else
        position_vector += b2
    end

    if layer == 2
        position_vector += [0, 0, h]
    end

    return SVector{3}(position_vector)
end


"""Constructor for the SiteIdx object"""
function SiteIdx(moire::SVector{2, Int}, graphene::SVector{2, Int}, sublattice::Int,
                 layer::Int, spin::Int, tbg_geom::TBGGeometry)::SiteIdx
    pos = get_position(moire, graphene, sublattice, layer, tbg_geom)
    return SiteIdx(moire, graphene, sublattice, layer, spin, pos)
end


function get_n_thresh(r_thresh::Float64, geom::TBGGeometry)::Int
    n_thresh = ceil(r_thresh / min(norm(geom.L1), norm(geom.L2)))
    return n_thresh
end


"""
Get the connection dictionary for each site in a moire cell (sites connected 
within the distance threshold). Note that they include themselves in the count.
Also, for all keys, the moire coordinate is set to (0,0) and spin = +1.
"""
function get_thresh_connections(r_thresh::Float64, n_thresh::Int,
                                geom::TBGGeometry)::Dict{SiteIdx, Vector{SiteIdx}}
    thresh_connections = Dict()
    
    # first looping through the origin graphene cells (keys) 
    for layer_orig in 1:2

        for g_orig in geom.graphene_cells[layer_orig]

            for sublattice_orig in 1:2
                orig_site = SiteIdx(SVector{2}(0, 0), g_orig, sublattice_orig, layer_orig, 1, geom)
                connected_sites = Vector()
                
                # then looping though all the cells in within the moire cells in the threshold (values)
                for moire_compare_1 in (-2 * n_thresh):(2 * n_thresh), moire_compare_2 in (-2 * n_thresh):(2 * n_thresh)
                    moire_compare = SVector{2}(moire_compare_1, moire_compare_2)

                    for layer_compare in 1:2

                        for g_compare in geom.graphene_cells[layer_compare]

                            for sublattice_compare in 1:2
                                compare_site = SiteIdx(moire_compare, g_compare, sublattice_compare, layer_compare, 1, geom)
                                r_separation = norm(orig_site.pos - compare_site.pos)

                                # if the comparison site falls within threshold, add it to the connected sites vector
                                if r_separation < r_thresh 
                                    push!(connected_sites, compare_site)
                                end
                            end
                        end 
                    end
                end

            thresh_connections[orig_site] = connected_sites
            end
        end
    end

    return thresh_connections
end


"""constructor for the TBGConstants object"""
function TBGConstants(geom::TBGGeometry)::TBGConstants

    t0 = 2.7
    t1 = 0.48
    q_sigma = 7.41
    q_pi = 3.15
    r_c = 5.0
    l_c = 0.265
    r_thresh = 8.0                                # need to think carefully about this value
    n_thresh = get_n_thresh(r_thresh, geom)       # number of moire cells within proximity (in L1/L2 directions, for a total of 4n^2 + 4n + 1)
    thresh_connections = get_thresh_connections(r_thresh, n_thresh, geom)

    return TBGConstants(t0, t1, q_sigma, q_pi, r_c, l_c, 
                        r_thresh, n_thresh, thresh_connections)               
end


# ===============================================================================
# Physics
# ===============================================================================

"""Get the coulomb interaction strength (accounts for possibility of same site interaction for opposite spins)"""
function get_V(si_μ::SiteIdx, si_v::SiteIdx)::Float64
    dr = norm(si_μ.pos - si_v.pos)
    return 1.438 / (0.116 + dr)     # in units of eV (took this form from arXiv:2511.04899v1)
end


"""Overarching container for all the important constants of the program"""
struct TBGSimulation 
    geometry::TBGGeometry
    constants::TBGConstants
    coupled_midxs::Dict{Int, Vector{Int}}         # midxs <=> matrix indices
    coupled_midxs_tunneling::Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64}}}
end


"""Dense 0-based index of a graphene cell within the moire cell (layer-dependent)"""
@inline function graphene_integer(g::SVector{2,Int}, layer::Int, mn::SVector{2,Int})::Int
    m, n = mn
    N = m^2 + m * n + n^2
    return layer == 1 ? mod(m * g[1] - n * g[2], N) : mod(n * g[1] - m * g[2], N)
end


"""Convert site index to matrix index"""
function si_to_mi(si::SiteIdx, geom::TBGGeometry)::Tuple{Int, SVector{3, Float64}}
    m, n = geom.mn
    N = m^2 + m * n + n^2

    g = graphene_integer(si.graphene, si.layer, geom.mn)
    sub = si.sublattice - 1
    layer = si.layer - 1
    spin = Int((1 - si.spin) / 2)             # spin +1 -> 0, -1 -> 1

    mi = 1 + g + N * (sub + 2 * layer + 4 * spin)        # g + N_g*s + N_g*N_s*l + N_g*N_s*N_l*σ
    R_moire = si.moire[1] * geom.L1 + si.moire[2] * geom.L2
    return mi, R_moire
end


"""function to construct coupled_midxs and coupled_midxs_tunneling dictionaries"""
function get_coupling_dicts(geom::TBGGeometry, consts::TBGConstants)::Tuple{Dict{Int, Vector{Int}}, Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64}}}}
    coupled_midxs = Dict{Int, Vector{Int}}()
    coupled_midxs_tunneling = Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64}}}()
    
    for (si_μ_0, connected_si_v) in consts.thresh_connections   # loop through key, value pairs
        # thresh_connections only has spin up sites, need to also do spin down
        for spin_μ in (1, -1)    # making sure to get both spin up and down
            si_μ = SiteIdx(si_μ_0.moire, si_μ_0.graphene, si_μ_0.sublattice, si_μ_0.layer, spin_μ, geom)
            mi_μ, _ = si_to_mi(si_μ, geom)
            unique_connected_mi_v = Set{Int}()    # need this for coupled_midxs

            for si_v_0 in connected_si_v
                for spin_v in (1, -1)      # making sure to get both spin up and down
                    si_v = SiteIdx(si_v_0.moire, si_v_0.graphene, si_v_0.sublattice, si_v_0.layer, spin_v, geom)
                    mi_v, η = si_to_mi(si_v, geom)
                    
                    push!(unique_connected_mi_v, mi_v)       # pushing to a set ensures that duplicates don't get added
                    V_μvη = get_V(si_μ, si_v)

                    mi_μv = SVector{2}(mi_μ, mi_v)

                    # adding onto coupled_midxs_tunneling
                    if haskey(coupled_midxs_tunneling, mi_μv)    # (μ, v) already in the dict
                        
                        push!(coupled_midxs_tunneling[mi_μv], (η, V_μvη))
                    else     # (μ, v) not yet in the dict
                        
                        coupled_midxs_tunneling[mi_μv] = [(η, V_μvη)]
                    end
                end
            end
            
            coupled_midxs[mi_μ] = collect(unique_connected_mi_v)
        end
    end

    return (coupled_midxs, coupled_midxs_tunneling)
end


"""constructor for the TBGSimulation object"""
function TBGSimulation(geom::TBGGeometry, consts::TBGConstants)
    coupled_midxs, coupled_midxs_tunneling = get_coupling_dicts(geom, consts)
    return TBGSimulation(geom, consts, coupled_midxs, coupled_midxs_tunneling)
end


# function to construct T(k) matrix (or all at once)
function construct_T_k()  # gotta leave it for tomorrow unfortunately :'(


# function to construct H(k)[{P(k):k}] (or all at once)


# functions relating to SCF HF


# plotting the resulting band structure




# ===============================================================================
# Testing/Benchmarking Functions
# ===============================================================================

"""
benchmarking tools and reminders:
- use @benchmark and display(_) to get the statistics of a function
- use @btime to get time and memory statistics (it prints automatically)
- use @time for time of a single run
- use @elapsed for only time 
"""


function plot_thresh_sites(m::Int, n::Int)::Nothing
    tbg_geom = TBGGeometry(m, n)
    tbg_consts = TBGConstants(tbg_geom)
    orig_site = SiteIdx(SVector(0, 0), SVector(0, 0), 1, 1, 1, tbg_geom)

    fig, ax = plt.subplots()

    # =================================================================
    # plot the entire lattice over top as well

    lattice_sites = []
    for layer in 1:2
        for moire_x in -1:1, moire_y in -1:1
            moire = SVector{2}(moire_x, moire_y)
            for graphene in tbg_geom.graphene_cells[layer]
                for sublattice in 1:2
                    site_idx = SiteIdx(moire, graphene, sublattice, layer, 1, tbg_geom)
                    site_pos = site_idx.pos
                    push!(lattice_sites, site_pos)
                end
            end
        end
    end

    xs_2 = [v[1] for v in lattice_sites]
    ys_2 = [v[2] for v in lattice_sites]

    ax.scatter(xs_2, ys_2, marker=".", alpha=0.1)

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


function test_thresh_construction_scaling()::Nothing
    times = []
    m_interval = 1:20
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
    plt.savefig("time for thresh construction.png")
end


# ===============================================================================
# User Area
# ===============================================================================


function main()
    println("=============== STARTING MAIN ===============")
    geom = TBGGeometry(18, 19)
    consts = TBGConstants(geom)
    println("moment of truth...")
    @btime TBGSimulation($geom, $consts)
    println("HORRAY, time to check...")
    plot_thresh_sites(8, 9)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
