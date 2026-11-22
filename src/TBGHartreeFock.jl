module TBGHartreeFock

include("useful/Useful.jl")
using .Useful
export progress, check_1rdm

include("geometry/Structure.jl")
using .Structure
export TBGGeometry, SiteIdx, TBGConstants

include("geometry/GeoMapping.jl")
using .GeoMapping
export TBGSimulation, mi_to_parts, si_to_mi, frac_to_cart, kidx_to_blockidx, spin_partition

include("hamiltonians/Hamiltonians.jl")
using .Hamiltonians
export construct_T_k, construct_Hartree_k, construct_Fock_k, get_HF_hamiltonian_k, hamiltonian_term_dict, HF_energy

include("optimization/Ansatze.jl")
using .Ansatze
export P_ansatz

include("optimization/SCFOptimizers.jl")
using .SCFOptimizers
export SCFOutput, SCF_optimize, plot_SCF_history, save_scf_output, load_scf_output

include("bands/BandStructure.jl")
using .BandStructure
export plot_TB_band_structure, plot_HF_band_structure, get_dos

include("visualization/Visualization.jl")
using .Visualization
export plot_thresh_sites, test_thresh_construction_scaling, visualize_1rdm_k_block, plot_magnetization, plot_density



end
