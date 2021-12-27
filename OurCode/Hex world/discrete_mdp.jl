include("./mdp.jl")
struct DiscreteMDP
    # TODO: Use sparse matrices?
    T::Array{Float64, 3} # T(s,a,s′)
    R::Array{Float64, 2} # R(s,a) = ∑_s' R(s,a,s')*T(s,a,s′)
    γ::Float64
end




