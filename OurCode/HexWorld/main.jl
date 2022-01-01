using Distributions
include("Policy.jl")
include("HexWorld.jl")
include("DiscreteMDP.jl")
include("MDP.jl")
include("Visualize.jl")

initHexWorld = HexWorld() #init data for HexWorld
mdp_hw = MDP(initHexWorld) #Construct MDP

#Policy evaluation area
k_max_iteration_policy_evaluation = 100
k_max_iteration_policy_value = 100

#policy evaluation U each state, whole length is the same as states
U_length = length(initHexWorld.hexes) + 1
U = [0.0 for i in 1:25]

#Policy Value function area
policy = ValueFunctionPolicy(mdp_hw, U)
#Policy generate action each state => Result 
policy_solution = solve(mdp_hw, policy, k_max_iteration_policy_value)

function policyAction()
    println("State => Action")
    for s = 1:24
        print(initHexWorld.hexes[s], " => ")
        println(policy_solution(s))
    end
end
policyAction()
visualizeResult()

