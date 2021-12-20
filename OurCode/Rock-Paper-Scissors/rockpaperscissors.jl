import Pkg

using JuMP
using LinearAlgebra

struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end

struct RockPaperScissors end

n_agents(simpleGame::RockPaperScissors) = 2

ordered_actions(simpleGame::RockPaperScissors, i::Int) = [:rock, :paper, :scissors]  # choose 1 in 3 actions
ordered_joint_actions(simpleGame::RockPaperScissors) = vec(collect(Iterators.product([ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)]...))) # create joint action space

n_joint_actions(simpleGame::RockPaperScissors) = length(ordered_joint_actions(simpleGame)) # amount of joint action space
n_actions(simpleGame::RockPaperScissors, i::Int) = length(ordered_actions(simpleGame, i)) # amount of actions

function reward(simpleGame::RockPaperScissors, i::Int, a)
    if i == 1
        noti = 2
    else
        noti = 1
    end

    if a[i] == a[noti]  # when draw, both reward = 0 
        r = 0.0
        # 6 cases when 1 win, 1 lose
        # reward of winner: 1.0, loser: -1.0
    elseif a[i] == :rock && a[noti] == :paper
        r = -1.0
    elseif a[i] == :rock && a[noti] == :scissors
        r = 1.0
    elseif a[i] == :paper && a[noti] == :rock
        r = 1.0
    elseif a[i] == :paper && a[noti] == :scissors
        r = -1.0
    elseif a[i] == :scissors && a[noti] == :rock
        r = -1.0
    elseif a[i] == :scissors && a[noti] == :paper
        r = 1.0
    end

    return r
end

function joint_reward(simpleGame::RockPaperScissors, a)
    # return vector U, U[i] is utility of agent i with joint action a
    return [reward(simpleGame, i, a) for i = 1:n_agents(simpleGame)]
end

function SimpleGame(simpleGame::RockPaperScissors)
    return SimpleGame(
        0.9,
        vec(collect(1:n_agents(simpleGame))),
        [ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)],
        (a) -> joint_reward(simpleGame, a)
    )
end

struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k, v) in zip(keys(p), vs))) # return SimpleGamePolicy from dictionary
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))  # return SimpleGamePolicy with probability of ai is 1
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)  # return probability agent i will do action ai

function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)  # return random action
end

joint(X) = vec(collect(Iterators.product(X...)))  # construct joint action space from X
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]  # replace π(i) with πi

function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i] * p(a) for a in joint(𝒜))  # the utility of agent i with joint policy π
end

function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    # loop value in 𝒫.𝒜[i] and find U[value] max
    ai = argmax(U, 𝒫.𝒜[i])
    return SimpleGamePolicy(ai)  # return deterministic best response with joint policy π
end

function softmax_response(𝒫::SimpleGame, π, i, λ)
    𝒜i = 𝒫.𝒜[i]
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    return SimpleGamePolicy(ai => exp(λ * U(ai)) for ai in 𝒜i)   # return softmax response, model how agent will select action ai
end

struct IteratedBestResponse
    k_max # number of iterations
    π # initial policy
end

# We use IteratedBestResponse because it MAY converge to Nash equilibrium
# Algorithm For Decision Making (page 495) 

function IteratedBestResponse(𝒫::SimpleGame, k_max)
    # k_max: how many times play
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
    return IteratedBestResponse(k_max, π)
end

function solve(M::IteratedBestResponse, 𝒫::SimpleGame)
    π = M.π
    for k = 1:M.k_max # loop k_max times
        # find best response with every agents
        π = [best_response(𝒫, π, i) for i in 𝒫.ℐ]
    end
    return π  # return policy (Nash equilibrium)
end


struct HierarchicalSoftmax
    λ # precision parameter
    k # level
    π # initial policy
end

function HierarchicalSoftmax(𝒫::SimpleGame, λ, k)
    # với mỗi joint action, vào từng action rồi gán dict value là 1
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]  # level k=0 is choosing action randomly
    return HierarchicalSoftmax(λ, k, π)
    # aims to model human agents, because people often do not play Nash equilibrium strategy
end

function solve(M::HierarchicalSoftmax, 𝒫)
    π = M.π
    for k = 1:M.k
        π = [softmax_response(𝒫, π, i, M.λ) for i in 𝒫.ℐ]
        # level k is a softmax response of level k-1
    end
    return π
end




mutable struct JointCorrelatedPolicy
    p # dictionary mapping from joint actions to probabilities
    JointCorrelatedPolicy(p::Base.Generator) = new(Dict(p))
end
(π::JointCorrelatedPolicy)(a) = get(π.p, a, 0.0)
function (π::JointCorrelatedPolicy)()
    D = SetCategorical(collect(keys(π.p)), collect(values(π.p)))
    return rand(D)
end


struct CorrelatedEquilibrium end
function solve(M::CorrelatedEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    model = Model(Ipopt.Optimizer)
    @variable(model, π[joint(𝒜)] ≥ 0)
    @objective(model, Max, sum(sum(π[a] * R(a) for a in joint(𝒜))))
    @constraint(model, [i = ℐ, ai = 𝒜[i], ai′ = 𝒜[i]],
        sum(R(a)[i] * π[a] for a in joint(𝒜) if a[i] == ai)
        ≥
        sum(R(joint(a, ai′, i))[i] * π[a] for a in joint(𝒜) if a[i] == ai))
    @constraint(model, sum(π) == 1)
    optimize!(model)
    return JointCorrelatedPolicy(a => value(π[a]) for a in joint(𝒜))
end

simpleGame = RockPaperScissors()
P = SimpleGame(simpleGame)



H = HierarchicalSoftmax(P, 0.3, 20) # H is used for finding policy for human agents
D = solve(H, P)

for i = 2:100
    print(i)
    print(": ")
    println(D[1].p[i])
end

