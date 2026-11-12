module TBGHartreeFock

include("geometry/Structure.jl")
using .Structure
export TBGGeometry, SiteIdx, TBGConstants

include("geometry/KspGeoMapping.jl")
using .KspGeoMapping
export TBGSimulation    # I don't think I need to export everythin the same as written in the module (as these are only for the user end)

include("hamiltonians/KspHamiltonians.jl")
using .KspHamiltonians
export construct_T_k, construct_Hartree_k, construct_Fock_k, get_HF_hamiltonian_k

include("optimizers/Ansatze.jl")
using .Ansatze
export P_ansatz, check_1rdm

include("optimizers/KspSCFOptimizers.jl")
using .KspSCFOptimizers
export SCFOutput, SCF_optimize, P_ansatz

include("bands/BandPlotting.jl")
using .BandPlotting
export plot_TB_band_structure, plot_HF_band_structure

include("visualization/Visualization.jl")
using .Visualization
export plot_thresh_sites, test_thresh_construction_scaling



end
