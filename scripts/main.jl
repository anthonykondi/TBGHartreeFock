using TBGHartreeFock
using StaticArrays, LinearAlgebra, Random
import PyPlot as plt

# USE: julia --project=. scripts/main.jl

function testing_scf()   # this way scopes don't get messed up
    sim = TBGSimulation(1, 2, N1=8, N2=8)
    rng = Random.seed!(1476)
    scfout = SCF_optimize(0.25, sim, rng=rng, kT=0.01; maxiter=100, oda=true, HF_type="GHF", ϵ_tol=1e-8)

    save_scf_output("nm=(1,2)_N1N2=(8,8)_quarter_filling_GHF_with_hubbard.jld2", scfout)
    # scfout = load_scf_output("nm=(3,4)_N1N2=(8,8)_half_filling_GHF_with_hubbard.jld2")

    # Checking P's spin structure
    P_local = local_density_matrix(scfout.P_dict)
    up, dn = spin_partition(sim)
    println("|P↑↑ - P↓↓| = ", norm(P_local[up, up] - P_local[dn, dn]))
    println("|P↑↓|       = ", norm(P_local[up, dn]))
    println("max |S_z|   = ", maximum(abs, site_observables(scfout.P_dict, sim).sz))
    println("|P↑↓| + |P↓↑| = ", norm(P_local[up, dn]) + norm(P_local[dn, up]))

    visualize_1rdm_k_block(scfout.P_dict[SVector{2}(0, 0)])

    N_e_actual = sim.N_tot
    N_e_calculated = 0
    for (kidx, P_k) in scfout.P_dict
        (h_err, p_err, e_num) = check_1rdm(P_k)
        N_e_calculated += round(Int, real(e_num))
        # println("kidx = $kidx:")
        # println("h_err = $h_err")
        # println("p_err = $p_err")
        # println("e_num = $e_num")
    end
    println("N_e_actual = $N_e_actual, N_e_calculated = $N_e_calculated")

    println("Plotting...")
    plot_HF_band_structure(scfout.E_dict, sim; tag="0.25 filling, unrestricted fock")

    dos = get_dos(scfout.E_dict)
    println("DOS integrated = $(sum(dos[:, 2]) * (dos[2, 1] - dos[1, 1]))") 

    plot_SCF_history(scfout)

    plot_density(scfout.P_dict, sim)
    plot_magnetization(scfout.P_dict, sim)

end


function visualizing()
    sim = TBGSimulation(3, 4, N1=8, N2=8)
    plot_thresh_sites(sim)
    scfout = load_scf_output("data/testing_scf/scf_outputs/nm=(1,2)_N1N2=(8,8)_quarter_filling_GHF.jld2")
    plot_density(scfout.P_dict, sim)
    plot_magnetization(scfout.P_dict, sim)
end


function testing_tb_plot()
    sim = TBGSimulation(4, 5, N1=15, N2=15)
    plot_TB_band_structure(sim, dos_η=0.05)
end

testing_scf()
