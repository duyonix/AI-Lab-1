struct POMG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    𝒪  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # joint reward function
end

function POMG(pomg::BabyPOMG)
    return POMG(
        pomg.babyPOMDP.γ, # discount factor
        vec(collect(1:n_agents(pomg))), # agents
        ordered_states(pomg), # state
        [ordered_actions(pomg, i) for i in 1:n_agents(pomg)], # joint action space
        [ordered_observations(pomg, i) for i in 1:n_agents(pomg)], # joint observation space
        (s, a, s′) -> transition(pomg, s, a, s′),  # Transition(s'|s, a)
        (a, s′, o) -> joint_observation(pomg, a, s′, o), # joint observation function
        (s, a) -> joint_reward(pomg, s, a) # Reward(s, a)
    )
end

function lookahead(𝒫::POMG, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    u′ = sum(T(s, a, s′) * sum(O(a, s′, o) * U(o, s′) for o in 𝒪) for s′ in 𝒮)
    return R(s, a) + γ * u′
end

function evaluate_plan(𝒫::POMG, π, s)
    # compute utility of conditional plan 
    a = Tuple(πi() for πi in π)
    U(o, s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π, o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s, a) : lookahead(𝒫, U, s, a) # equation (26.1) page 528
end

function utility(𝒫::POMG, b, π)
    # compute utility of policy π from initial state distibution b
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u)) # equation (26.2) page 528
end