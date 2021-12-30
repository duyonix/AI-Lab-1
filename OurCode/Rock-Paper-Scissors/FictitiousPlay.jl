using Pkg

using LinearAlgebra
using GridInterpolations
using CategoricalArrays
using Distributions
using Random
using JuMP
using Ipopt
using StatsPlots
using DataFrames

# using IndexedTables
struct SimpleGame
    γ  # discount factor
    ℐ  # agents
    𝒜  # joint action space
    R  # joint reward function
end

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

# Policy là 1 dictionary action-probability
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    # create struct SimpleGamePolicy containing 1 dictionary
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        # trả về SimpleGamePolicy từ dictionary, được tính là action-probability
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))  # return SimpleGamePolicy với probability của action ai là 1.0
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)  # return probability agent i sẽ thực hiện action ai

struct SetCategorical{S}
    elements::Vector{S} # Set elements (could be repeated)
    distr::Categorical # Categorical distribution over set elements


    # normalize: convert to 0->1 and keep the ratio

    # norm(arr, type 1|2): 
    # type 1: sum of absolute value
    # type 2: norm(arr) sqrt (sum of square)
    function SetCategorical(elements::AbstractVector{S}) where {S}
        print("SetCategorical\n")

        weights = ones(length(elements)) # initialize weight of each element is 1
        return new{S}(elements, Categorical(normalize(weights, 1))) # convert to propability
    end


    function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where {S}

        ℓ₁ = norm(weights, 1)
        # not exist any element (= 0 all)
        if ℓ₁ < 1e-6 || isinf(ℓ₁)
            return SetCategorical(elements) # first time
        end
        # normalize the weight to probability for each e (0->1)
        distr = Categorical(normalize(weights, 1))
        return new{S}(elements, distr)
    end
end

# over load
Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]

Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]

function Distributions.pdf(D::SetCategorical, x)
    # zip: pack to a pair
    # sum = distr.p of x in D.elements, if not x => 0
    sum(e == x ? w : 0.0 for (e, w) in zip(D.elements, D.distr.p))
end

function (πi::SimpleGamePolicy)()
    # from 2 arrays: keys + values => 2 arr keys + propability (weights converted to 0->1)
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)  # return random action
end

joint(X) = vec(collect(Iterators.product(X...)))  # create joint action space from X
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]  # replace π[i] with πi in π

function utility(𝒫::SimpleGame, π, i)

    𝒜, R = 𝒫.𝒜, 𝒫.R
    # probability action a occur
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    # U: evaluate the practicality of agent i's policy 

    return sum(R(a)[i] * p(a) for a in joint(𝒜))  # the utility of agent i with joint policy π
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

    function VisualizeRPS(k_max)
        model = [DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max)),
            DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max))]
        policy = [DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max)),
            DataFrame(rock = zeros(k_max), paper = zeros(k_max), scissors = zeros(k_max))]
        return new(model, policy)
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

# iteration: 100
k_max = 100
v, s = simulate(P, pi, k_max)
display(s)
# visualize
model1 = @df v.model[1] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], xlabel = "iteration", title = "opponent model (agent 1)")
model2 = @df v.model[2] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], title = "opponent model (agent 2)")
policy1 = @df v.policy[1] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], legend = false, title = "policy agent 1")
policy2 = @df v.policy[2] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], legend = false, xlabel = "iteration", title = "policy agent 2")

plot(model2, policy1, model1, policy2, layout = (2, 2), size = (900, 700), grid = :off, ylim = (-0.05, 1))
