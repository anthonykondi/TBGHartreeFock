module Useful
using LinearAlgebra, SparseArrays
export progress, check_1rdm


"""Helpful progress tally counter"""
function progress(message::String, iter::Int, maxiter::Int)
    print("\r$(message): $iter / $maxiter")
    flush(stdout)
    iter == maxiter && println()
end


"""Print check of 1rdm properties"""
function check_1rdm(P::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}})     
    h_err = norm(P - P')
    p_err = norm(P - P * P)
    e_num = tr(P)

    return (h_err, p_err, e_num)
end


end