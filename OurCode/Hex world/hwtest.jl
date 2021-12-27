using Distributions
include("policy.jl")
include("hexworld.jl")
include("discrete_mdp.jl")
include("mdp.jl")
m=HexWorld()
mdp_hw=MDP(m)

#Policy evaluation area
k_max_iteration_policy_evaluation=5
k_max_iteration_policy_value=100

#policy evaluation U each state, whole length is the same as states
U_length=length(m.hexes)+1
U=[0.0 for i in 1:25]

#Policy Value function area
policy=ValueFunctionPolicy(mdp_hw,U)

policy_solution=solve(mdp_hw,policy,k_max_iteration_policy_value)

function policyAction()
    for s = 1:25
        print(s)
        println(policy_solution(s))
    end
end
# policyAction()
# display(mdp_hw.T)

# function checkX(T,𝒮)
#     for s′ in 𝒮
#         println(s′)
#         println(T[1,2,s′])
#     end
# end

# checkX(mdp_hw.T,mdp_hw.𝒮)