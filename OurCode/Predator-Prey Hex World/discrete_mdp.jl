struct DiscreteMDP
    T::Array{Float64,3} # T(s,a,s′)
    R::Array{Float64,2} # R(s,a) = ∑_s' R(s,a,s')*T(s,a,s′)
    γ::Float64
end

n_states(mdp::DiscreteMDP) = size(mdp.T, 1)
n_actions(mdp::DiscreteMDP) = size(mdp.T, 2)
discount(mdp::DiscreteMDP) = mdp.γ
ordered_states(mdp::DiscreteMDP) = collect(1:n_states(mdp))
ordered_actions(mdp::DiscreteMDP) = collect(1:n_actions(mdp))
state_index(mdp::DiscreteMDP, s::Int) = s


