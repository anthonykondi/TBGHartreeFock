module KspGeoMapping
export TBGSimulation, mi_to_parts, si_to_mi, frac_to_cart

using LinearAlgebra, StaticArrays

using ..Structure


"""Overarching container for all the important constants of the program"""
struct TBGSimulation 
    geometry::TBGGeometry
    constants::TBGConstants
    coupled_midxs::Dict{Int, Vector{Int}}         # midxs = matrix indices
    coupled_midxs_tunneling::Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64, Float64}}}
    moire_N::SVector{2, Int}                      # (N1, N2) moire unit cells
end


"""Get the coulomb interaction strength (accounts for possibility of same site interaction for opposite spins)"""
function get_V_strength(si_μ::SiteIdx, si_v::SiteIdx)::Float64
    dr = norm(si_μ.pos - si_v.pos)
    return 1.438 / (0.116 + dr)     # in units of eV (took this form from arXiv:2511.04899v1)
end


"""Get the hopping strength"""
function get_t_strength(si_μ::SiteIdx, si_v::SiteIdx, geom::TBGGeometry, consts::TBGConstants)::Float64
    
    if si_μ == si_v    # same site special case (onsite energy of the p_z orbital at site i)
        return 0.0     # setting to 0 for now (simplest choice)
    end

    if si_μ.spin != si_v.spin  # Hopping conserves spin
        return 0.0
    end

    r = norm(si_μ.pos - si_v.pos)
    n = dot([0, 0, 1], (si_μ.pos - si_v.pos) / r)

    F_c = 1 / (1 + exp((r - consts.r_c) / consts.l_c))
    V_ppπ = -consts.t0 * exp(consts.q_pi * (1 - r / geom.d)) * F_c
    V_ppsigma = consts.t1 * exp(consts.q_sigma * (1 - r / geom.h)) * F_c

    return n^2 * V_ppsigma + (1 - n^2) * V_ppπ
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


"""Decompose matrix index into (g, sublattice, layer, spin_bit), all 0-based"""
function mi_to_parts(mi::Int, geom::TBGGeometry)::NTuple{4, Int}
    m, n = geom.mn
    N = m^2 + m * n + n^2

    rest, g = divrem(mi - 1, N)        # g in 0:N-1
    rest, sub = divrem(rest, 2)        # sub in 0:1
    spin, layer = divrem(rest, 2)      # layer in 0:1, spin in 0:1
    return g, sub, layer, spin         # more efficient than converting to SiteIdx
end


"""function to construct coupled_midxs and coupled_midxs_tunneling dictionaries"""
function get_coupling_dicts(geom::TBGGeometry, consts::TBGConstants)::Tuple{Dict{Int, Vector{Int}}, Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64, Float64}}}}
    coupled_midxs = Dict{Int, Vector{Int}}()
    coupled_midxs_tunneling = Dict{SVector{2, Int}, Vector{Tuple{SVector{3, Float64}, Float64, Float64}}}()
    
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
                    
                    V_μvη = get_V_strength(si_μ, si_v)
                    t_μvη = get_t_strength(si_μ, si_v, geom, consts)

                    mi_μv = SVector{2}(mi_μ, mi_v)

                    # adding onto coupled_midxs_tunneling
                    if haskey(coupled_midxs_tunneling, mi_μv)    # (μ, v) already in the dict

                        push!(coupled_midxs_tunneling[mi_μv], (η, V_μvη, t_μvη))
                    else     # (μ, v) not yet in the dict
                        
                        coupled_midxs_tunneling[mi_μv] = [(η, V_μvη, t_μvη)]
                    end
                end
            end
            
            coupled_midxs[mi_μ] = collect(unique_connected_mi_v)
        end
    end

    return (coupled_midxs, coupled_midxs_tunneling)
end


"""constructor for the TBGSimulation object from the geometry, consts, and moire_N"""
function TBGSimulation(geom::TBGGeometry, consts::TBGConstants, moire_N::SVector{2, Int})::TBGSimulation
    println("Getting coupling_dicts...")
    @time coupled_midxs, coupled_midxs_tunneling = get_coupling_dicts(geom, consts)
    return TBGSimulation(geom, consts, coupled_midxs, coupled_midxs_tunneling, moire_N)
end


"""constructor for the TBGSimulation object from mn and moire_N (optional, in case you only need to do tight binding)"""
function TBGSimulation(m::Int, n::Int; r_thresh::Float64 = 8.0, N1::Int = 1, N2::Int = 1)::TBGSimulation
    geom = TBGGeometry(m, n)
    consts = TBGConstants(geom; r_thresh)   # don't forget keyword arg
    moire_N = SVector{2}(N1, N2)
    return TBGSimulation(geom, consts, moire_N)
end


"""Get k from its fractional coordinates"""
frac_to_cart(f1, f2, geo) = SVector{3}(f1 * geo.k1 + f2 * geo.k2)



end