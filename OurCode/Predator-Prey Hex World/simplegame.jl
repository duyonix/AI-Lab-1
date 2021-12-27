struct SimpleGame
    γ  # discount factor
    ℐ  # agents
    𝒜  # joint action space
    R  # joint reward function
end

using LinearAlgebra
using GridInterpolations
using CategoricalArrays
using Distributions
using Random



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