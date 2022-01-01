include("./mdp.jl")

#Struct Discrete Markov Decision Process
struct DiscreteMDP
    T::Array{Float64, 3} # transition function
    R::Array{Float64, 2}  #reward function
    γ::Float64 # discount factor
end




