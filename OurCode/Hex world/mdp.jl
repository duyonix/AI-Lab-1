struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
    TR # sample transition and reward
end

function MDP(T::Array{Float64, 3}, R::Array{Float64, 2}, γ::Float64)
    MDP(γ, 1:size(R,1), 1:size(R,2), T, R, nothing)
end
