import Pkg

using JuMP
using LinearAlgebra

struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end

struct Travelers end

n_agents(simpleGame::Travelers) = 2

ordered_actions(simpleGame::Travelers, i::Int) = 2:100  # each traveler has to choose 1 integer from 2 to 100
ordered_joint_actions(simpleGame::Travelers) = vec(collect(Iterators.product([ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)]...)))
# Vector of ordered actions

n_joint_actions(simpleGame::Travelers) = length(ordered_joint_actions(simpleGame))
n_actions(simpleGame::Travelers, i::Int) = length(ordered_actions(simpleGame, i))

function reward(simpleGame::Travelers, i::Int, a)
    if i == 1
        noti = 2
    else
        noti = 1
    end
    if a[i] == a[noti]
        r = a[i]
    elseif a[i] < a[noti]
        r = a[i] + 2
    else
        r = a[noti] - 2
    end
    return r
end

function joint_reward(simpleGame::Travelers, a)
    # return vector U, U[i] is utility of agent i with joint action a
    return [reward(simpleGame, i, a) for i in 1:n_agents(simpleGame)]
end

function SimpleGame(simpleGame::Travelers)
    return SimpleGame(
        0.9,
        vec(collect(1:n_agents(simpleGame))),
        [ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)],
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
        return new(Dict(k => v for (k,v) in zip(keys(p), vs))) # return SimpleGamePolicy from dictionary
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
    return sum(R(a)[i]*p(a) for a in joint(𝒜))  # the utility of agent i with joint policy π
end

function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜[i])
    return SimpleGamePolicy(ai)  # return deterministic best response with joint policy π
end

function softmax_response(𝒫::SimpleGame, π, i, λ)
    𝒜i = 𝒫.𝒜[i]
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    return SimpleGamePolicy(ai => exp(λ*U(ai)) for ai in 𝒜i)   # return softmax response, model how agent will select action ai
end

struct IteratedBestResponse
    k_max # number of iterations
    π # initial policy
end

# We use IteratedBestResponse because it MAY converge to Nash equilibrium
# Algorithm For Decision Making (page 495) 

function IteratedBestResponse(𝒫::SimpleGame, k_max)
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
    return IteratedBestResponse(k_max, π)
end
    
function solve(M::IteratedBestResponse, 𝒫::SimpleGame)
    π = M.π
    for k in 1:M.k_max
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
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]  # level k=0 is choosing action randomly
    return HierarchicalSoftmax(λ, k, π)  
    # aims to model human agents, because people often do not play Nash equilibrium strategy
end

function solve(M::HierarchicalSoftmax, 𝒫)
    π = M.π
    for k in 1:M.k
        π = [softmax_response(𝒫, π, i, M.λ) for i in 𝒫.ℐ]
        # level k is a softmax response of level k-1
    end
    return π
end


simpleGame=Travelers()  # simpleGame::Travelers
P=SimpleGame(simpleGame) # P is a SimpleGame instance according to simpleGame
M=IteratedBestResponse(P, 100) # M is used for finding Nash Equilibrium
H=HierarchicalSoftmax(P, 0.3, 5) # H is used for finding policy for human agents
D=solve(H, P)

for i in 2:100
    print(i)
    print(": ")
    println(D[1].p[i])
end
