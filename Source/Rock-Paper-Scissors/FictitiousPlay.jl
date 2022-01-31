using Pkg

using LinearAlgebra
using GridInterpolations
using CategoricalArrays
using Distributions
using Random
using JuMP
using Ipopt
using DataFrames

include("../helpers/SimpleGame/SimpleGame.jl")
include("Visualize.jl")


struct RockPaperScissors end

# 2 agents
n_agents(simpleGame::RockPaperScissors) = 2
# 3 actions
ordered_actions(simpleGame::RockPaperScissors, i::Int) = [:rock, :paper, :scissors]
# create array tuple 2 actions => joint action space
ordered_joint_actions(simpleGame::RockPaperScissors) = vec(collect(Iterators.product([ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)]...)))
n_joint_actions(simpleGame::RockPaperScissors) = length(ordered_joint_actions(simpleGame))
n_actions(simpleGame::RockPaperScissors, i::Int) = length(ordered_actions(simpleGame, i))

# reward function
function reward(simpleGame::RockPaperScissors, i::Int, a)
    if i == 1
        noti = 2
    else
        noti = 1
    end

    if a[i] == a[noti]
        r = 0.0
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

# 2 reward of a joint action tuple
function joint_reward(simpleGame::RockPaperScissors, a)
    return [reward(simpleGame, i, a) for i = 1:n_agents(simpleGame)]
end
# initialize Simple Game for RockPaperScissors 
function SimpleGame(simpleGame::RockPaperScissors)
    return SimpleGame(
        0.9,
        vec(collect(1:n_agents(simpleGame))),
        [ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)],
        (a) -> joint_reward(simpleGame, a)
    )
end


function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜[i])
    return SimpleGamePolicy(ai)  # return deterministic best response with joint policy π
end


# visualize
struct VisualizeRPS
    model
    policy
    rewards

    function VisualizeRPS(k_max)
        model = [DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max)),
            DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max))]
        policy = [DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max)),
            DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max))]
        rewards = Vector{Tuple{Int64,Int64}}()
        return new(model, policy, rewards)
    end
end



function simulate(𝒫::SimpleGame, π, k_max)

    v = VisualizeRPS(k_max)
    # round 1: model => 1/3
    for k = 1:k_max
        # return random action from (πi::SimpleGamePolicy)()
        a = [πi() for πi in π]

        for πi in π
            update!(πi, a, v, k)
        end

        # update reward visualize
        reward = 𝒫.R(a)
        if (k > 1)
            reward[1] += v.rewards[k-1][1]
            reward[2] += v.rewards[k-1][2]
        end
        push!(v.rewards, Tuple(reward))
    end

    return v, π
end

mutable struct FictitiousPlay
    𝒫 # simple game
    i # agent index
    N # array of action count dictionaries => 2 Dict, every Dict 3 actions corresponding to 3 counts => calculate policy(result)
    # Save the opponent's counts to decide what next policy
    πi # current policy => only 1 currently selected action
end
function FictitiousPlay(𝒫::SimpleGame, i)
    # arrays consist of 2 Dict of 2 agents
    # initialize the number of times each action is 1 (counts)
    N = [Dict(aj => 1 for aj in 𝒫.𝒜[j]) for j in 𝒫.ℐ]
    πi = SimpleGamePolicy(ai => 1.0 / 3 for ai in 𝒫.𝒜[i])
    return FictitiousPlay(𝒫, i, N, πi)
end
# current policy (dict)
(πi::FictitiousPlay)() = πi.πi()
# probability of each actions
(πi::FictitiousPlay)(ai) = πi.πi(ai)

function update!(πi::FictitiousPlay, a, v, iteration)
    N, 𝒫, ℐ, i = πi.N, πi.𝒫, πi.𝒫.ℐ, πi.i

    # function to calculate policy của agent j
    p(j) = SimpleGamePolicy(aj => u / sum(values(N[j])) for (aj, u) in N[j])
    # update visualize policy
    v.policy[i][iteration, a[i]] = 1


    # update visualize model
    v.model[i][iteration, :rock] = p(i).p[:rock]
    v.model[i][iteration, :paper] = p(i).p[:paper]
    v.model[i][iteration, :scissors] = p(i).p[:scissors]

    for (j, aj) in enumerate(a)
        N[j][aj] += 1 # agent j with action aj +=1 count
    end
    # display(p(1))
    # array 2 policy of 2 agents
    π = [p(j) for j in ℐ]
    # get policy (action-probability) of agent i & opponent, policy is calculated from probability (calculated from N), then use best_response to solve and return πi.πi is the selected action (choose)
    # update current policy
    πi.πi = best_response(𝒫, π, i)

end


# -----------RUN------------------------
# Initialize Simple Game RockPaperScissors P
simpleGame = RockPaperScissors()
P = SimpleGame(simpleGame)

# Initialize FictitiousPlay for each agent
pi = [(FictitiousPlay(P, i)) for i in 1:2]

# iteration: 1000000

k_max = 1000000

v, s = simulate(P, pi, k_max)
display(s)

# visualize
visualizeRPS(v)

