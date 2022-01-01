include("MDP.jl")

#Struct Discrete Markov Decision Process
struct DiscreteMDP
    T::Array{Float64, 3} # transition function
    R::Array{Float64, 2}  #reward function
    γ::Float64 # discount factor
end


n_states(mdp::DiscreteMDP) = size(mdp.T, 1)
n_actions(mdp::DiscreteMDP) = size(mdp.T, 2)
discount(mdp::DiscreteMDP) = mdp.γ
ordered_states(mdp::DiscreteMDP) = collect(1:n_states(mdp))
ordered_actions(mdp::DiscreteMDP) = collect(1:n_actions(mdp))
state_index(mdp::DiscreteMDP, s::Int) = s

