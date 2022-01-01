#Struct Markov Decision Process

struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
end

#Constructor MDP 
function MDP(T::Array{Float64, 3}, R::Array{Float64, 2}, γ::Float64)
    MDP(γ, 1:size(R,1), 1:size(R,2), T, R)
end
