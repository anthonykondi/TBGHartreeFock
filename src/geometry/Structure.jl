module Structure
export TBGGeometry, SiteIdx, TBGConstants

using LinearAlgebra, StaticArrays


"""Geometrical factors and quantities stored in a common place"""
struct TBGGeometry
    d::Float64
    h::Float64 
    mn::SVector{2, Int}
    N_tot::Int
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
    kΓ::SVector{3, Float64}
    kM::SVector{3, Float64}
    kK::SVector{3, Float64}
    kKp::SVector{3, Float64}
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


"""Other required constants stored in a common place"""
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
    layer_1_graphene = SVector{2, Int}[]
    for i in -R:R, j in 0:2*R
        s1 = (n + m) * i + m * j
        s2 = -m * i + n * j
        if 0 <= s1 < N && 0 <= s2 < N
            push!(layer_1_graphene, SVector{2}(i, j))
        end
    end
    graphene_in_moire[1] = layer_1_graphene

    # layer 2
    layer_2_graphene = SVector{2, Int}[]
    for i in -R:R, j in 0:2*R
        s1 = (m + n) * i + n * j
        s2 = -n * i + m * j
        if 0 <= s1 < N && 0 <= s2 < N
            push!(layer_2_graphene, SVector{2}(i, j))
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
    N_tot = 8 * (n^2 + n*m + m^2)
    println("Getting graphene cells...")
    @time graphene_cells = get_graphene_cells(mn)
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

    # reciprocal lattice vectors of the supercell spanned by (L1, L2)
    A = 2 * pi / (L1[1] * L2[2] - L1[2] * L2[1])
    k1 = SVector{3}(A * [L2[2], -L2[1], 0])
    k2 = SVector{3}(A * [-L1[2], L1[1], 0])

    # high symmetry points
    kΓ  = SVector{3}(0.0, 0.0, 0.0)
    kM  = SVector{3}(k1 / 2)
    kK  = SVector{3}((k1 + 2 * k2) / 3)
    kKp = SVector{3}((2 * k1 + k2) / 3)

    return TBGGeometry(d, h, mn, N_tot, graphene_cells, θ, a1_down,a1_up, a2_down, a2_up, b1_down, b1_up, b2_down, 
                        b2_up, L1, L2, k1, k2, kΓ, kM, kK, kKp)            
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


"""Minimum number of moire unit cells to check in each direction"""
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
    thresh_connections = Dict{SiteIdx, Vector{SiteIdx}}()
    
    # first looping through the origin graphene cells (keys) 
    for layer_orig in 1:2

        for g_orig in geom.graphene_cells[layer_orig]

            for sublattice_orig in 1:2
                orig_site = SiteIdx(SVector{2}(0, 0), g_orig, sublattice_orig, layer_orig, 1, geom)
                connected_sites = SiteIdx[]
                
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
function TBGConstants(geom::TBGGeometry; r_thresh::Float64 = 8.0)::TBGConstants

    t0 = 2.7
    t1 = 0.48
    q_sigma = 7.41
    q_pi = 3.15
    r_c = 5.0
    l_c = 0.265
    n_thresh = get_n_thresh(r_thresh, geom)       # number of moire cells within proximity (in L1/L2 directions, for a total of 4n^2 + 4n + 1)
    println("Getting thresh_connections...")
    @time thresh_connections = get_thresh_connections(r_thresh, n_thresh, geom)

    return TBGConstants(t0, t1, q_sigma, q_pi, r_c, l_c, 
                        r_thresh, n_thresh, thresh_connections)               
end



end