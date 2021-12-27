# Our code is based on the book "Algorithm for Decision Making" by Mykel J. Kochenderfer, Tim A. Wheeler, Kyle Wray
# from The MIT Press; Cambridge, Massachusetts; London, England

import Pkg
import JuMP
import LinearAlgebra
using Plots

# Model SimpleGame is a fundamental model for multiagent reasoning
struct SimpleGame
    γ   # discount factor
    ℐ   # agents
    𝒜   # joint action spaces
    R   # joint reward function
end

struct Travelers end    # Model for Game Theory: Travelers Dilemma 

n_agents(simpleGame::Travelers) = 2 # represented for number of agents in the game

ordered_actions(simpleGame::Travelers, i::Int) = 2:100  # each traveler has to choose 1 integer from 2 to 100

# Vector of ordered actions for n_agents(simpleGame::Travelers)
ordered_joint_actions(simpleGame::Travelers) = vec(
    collect(Iterators.product([ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)]...))
)

n_joint_actions(simpleGame::Travelers) = length(ordered_joint_actions(simpleGame))  # number of joint actions
n_actions(simpleGame::Travelers, i::Int) = length(ordered_actions(simpleGame, i))   # number of actions for each agent

# function to produce reward for traveler i in the game
function reward(simpleGame::Travelers, i::Int, a)
    if i == 1
        noti = 2    # the other traveler
    else
        noti = 1
    end
    if a[i] == a[noti]      # two agents choose the same money
        r = a[i]
    elseif a[i] < a[noti]   # traveler i gets less money than the other traveler
        r = a[i] + 2
    else                    # traveler i gets more money than the other traveler
        r = a[noti] - 2
    end
    return r    # true reward for agent i
end

# joint reward function for all agents in the game
function joint_reward(simpleGame::Travelers, a)
    return [reward(simpleGame, i, a) for i = 1:n_agents(simpleGame)]
end

# construct SimpleGame for Travelers Dilemma problem
function SimpleGame(simpleGame::Travelers)
    return SimpleGame(
        0.9,    # default discount factor for Travelers Dilemma is 0.9
        vec(collect(1:n_agents(simpleGame))),
        [ordered_actions(simpleGame, i) for i = 1:n_agents(simpleGame)],
        (a) -> joint_reward(simpleGame, a)
    )
end

# Model Policy for Simple Game
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

# function to  compute the utility associated with executing joint policy π in the game 𝒫 from the perspective of agent i.
function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i] * p(a) for a in joint(𝒜))  # the utility of agent i with joint policy π
end

# A best response of agent i to the policies of the other agents π^(−i) is a policy π^i 
# that maximizes the utility of the game 𝒫 from the perspective of agent i.
# The formula for best response below is in page 495 of the book
#                       Ui(π^i, π^(−i)) ≥ Ui(π^(i′), π^(−i))
function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜[i])  # maximizes the utility 
    return SimpleGamePolicy(ai)
end

# A softmax response e to model how agent i will select their action with the precision parameter λ
# We often use softmax response to calculate how people will do their actions in the game
# precision parameter λ is a probability in thinking of people to be more confident in their actions
# The formula for softmax response below is in page 497 of the book
#                       πi(ai) ∝ exp(λUi(ai, π−i)) 
function softmax_response(𝒫::SimpleGame, π, i, λ)
    𝒜i = 𝒫.𝒜[i]
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    return SimpleGamePolicy(ai => exp(λ * U(ai)) for ai in 𝒜i)
end

# experiment 1: calculate the reward of 2 computer agents
# the result will come close to Nash Equilibrium of the Travelers Dilemma: --> $2

# Model Iterated Best Response (page 503)
struct IteratedBestResponse
    k_max   # number of iterations
    π       # initial policy
end

# constructor that takes as input a simple game and creates an initial policy that has each agent select actions uniformly at random
function IteratedBestResponse(𝒫::SimpleGame, k_max)
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
    return IteratedBestResponse(k_max, π)
end

# function to solve Iterated Best Response 
function solve(M::IteratedBestResponse, 𝒫::SimpleGame)
    π = M.π
    for k = 1:M.k_max
        # use the best response to update the policy of each agent
        π = [best_response(𝒫, π, i) for i in 𝒫.ℐ]
    end
    return π  # return policy, often close to Nash Equilibrium
end

# experiment 2: calculate the reward of 2 humen agents (behavioral game theory)
# the result we expect tend to be between $97 and $100 for human agents

# Model Hierarchical Softmax (page 504)
struct HierarchicalSoftmax
    λ # precision parameter
    k # level
    π # initial policy
end

# By default, it starts with an initial joint policy that assigns uniform probability to all individual actions.
function HierarchicalSoftmax(𝒫::SimpleGame, λ, k)
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
    return HierarchicalSoftmax(λ, k, π)
    # the result aims to model human agents, because people often do not play Nash equilibrium strategy
end

# function to solve Hierarchical Softmax 
function solve(M::HierarchicalSoftmax, 𝒫)
    π = M.π
    for k = 1:M.k
        # use the softmax response to update the policy of each agent with the precision parameter λ 
        π = [softmax_response(𝒫, π, i, M.λ) for i in 𝒫.ℐ]
    end
    return π
end


# EXAMPLE for our experiment

simpleGame = Travelers()  # simpleGame::Travelers
P = SimpleGame(simpleGame) # P is a SimpleGame instance according to simpleGame

# example of experiment 1: Iterated Best Response
# IBR = IteratedBestResponse(P, 100) # IBR is used for finding policy for computer agents
# π1 = solve(IBR, P)
# print(π1)

# example of experiment 2: Hierarchical Softmax
# run the code below in REPL to see the visualization
HS = HierarchicalSoftmax(P, 0.5, 10) # HS is used for finding policy for human agents
π2 = solve(HS, P)

# visualize result with Plots
bar(collect(keys(π2[1].p)), collect(values(π2[1].p)), orientation = :vertical, legend = false)

# show the result to console
# for i = 2:100
#     print(i)
#     print(": ")
#     println(π2[1].p[i])
# end
