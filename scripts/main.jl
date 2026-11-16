using TBGHartreeFock
using StaticArrays, LinearAlgebra
import PyPlot as plt

# USE: julia --project=. scripts/main.jl

function testing_scf()   # this way scopes don't get messed up
    sim = TBGSimulation(1, 2, N1=4, N2=4)
    scfout = SCF_optimize(0.5, sim)

    println(maximum(norm(H - H') for H in values(scfout.H_dict)))

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
    plot_HF_band_structure(scfout.E_dict, sim; tag="with_interaction_back", dos_η=0.1)

    dos = get_dos(scfout.E_dict)
    println("DOS integrated = $(sum(dos[:, 2]) * (dos[2, 1] - dos[1, 1]))") 

    plot_SCF_history(scfout)

end


function testing_tb_plot()
    sim = TBGSimulation(4, 5, N1=15, N2=15)
    plot_TB_band_structure(sim, dos_η=0.05)
end

testing_scf()

