using Pkg

using Distributions
struct MG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    T  # transition function
    R  # joint reward function
end


include("discrete_mdp.jl")
include("hexworld.jl")
include("simplegame.jl")
include("helper.jl")
include("mg.jl")
include("grid.jl")

using Random
using JuMP
using Distributions
using CategoricalArrays
using LinearAlgebra
using GridInterpolations
using DataFrames
using StatsPlots
using IndexedTables



struct PredatorPreyHexWorldMG
    hexes::Vector{Tuple{Int,Int}}
    hexWorldDiscreteMDP::DiscreteMDP
end

n_agents(mg::PredatorPreyHexWorldMG) = 2

ordered_states(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:length(mg.hexes)))
ordered_states(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_states(mg, i) for i in 1:n_agents(mg)]...)))

ordered_actions(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:n_actions(mg.hexWorldDiscreteMDP)))
ordered_joint_actions(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_actions(mg, i) for i in 1:n_agents(mg)]...)))

n_actions(mg::PredatorPreyHexWorldMG, i::Int) = length(ordered_actions(mg, i))
n_joint_actions(mg::PredatorPreyHexWorldMG) = length(ordered_joint_actions(mg))

function transition(mg::PredatorPreyHexWorldMG, s, a, s′)

    # Khi prey bị bắt, prey mới sẽ được sinh ra, rồi dịch chuyển ngẫu nhiên tới một vị trí nào đó trên hex map nên tỉ lệ sẽ là 1/12. Còn predator sẽ đứng yên
    if s[1] == s[2]
        prob = Float64(s′[1] == s[1]) / length(mg.hexes)
        #display(prob)
    else


        # display(mg.hexWorldDiscreteMDP.T[:, :, 1])

        # Ngược lại, transition cả 2 sẽ theo HexWorld
        # Vì cố 2 agents nên nhân lại
        prob = mg.hexWorldDiscreteMDP.T[s[1], a[1], s′[1]] * mg.hexWorldDiscreteMDP.T[s[2], a[2], s′[2]]
    end
    # xác suất transition của cả 2 agents
    return prob
end

function reward(mg::PredatorPreyHexWorldMG, i::Int, s,a)
    r = 0.0

    if i == 1
        # Predators get -1 for moving and 10 for catching the prey.
        if s[1] == s[2]
            return 10.0
        else
            return -1.0
        end
    elseif i == 2
        # Prey get -1 for moving and -100 for being caught.
        if s[1] == s[2]
            r = -100.0
        else
            r = -1.0
        end
    end

    return r
end

function joint_reward(mg::PredatorPreyHexWorldMG, s, a)
    return [reward(mg, i, s, a) for i in 1:n_agents(mg)]
end

function MG(mg::PredatorPreyHexWorldMG)
    return MG(
        mg.hexWorldDiscreteMDP.γ,
        vec(collect(1:n_agents(mg))),
        ordered_states(mg),
        [ordered_actions(mg, i) for i in 1:n_agents(mg)],
        (s, a, s′) -> transition(mg, s, a, s′),
        (s,a) -> joint_reward(mg, s,a)
    )
end

function PredatorPreyHexWorldMG(hexes::Vector{Tuple{Int,Int}},
    r_bump_border::Float64,
    p_intended::Float64,
    γ::Float64)
    hexWorld = HexWorldMDP(hexes,
        r_bump_border,
        p_intended,
        Dict{Tuple{Int64,Int64},Float64}(),
        γ)
    mdp = hexWorld.mdp
    return PredatorPreyHexWorldMG(hexes, mdp)
end

struct VisualizePPHW
    model
    policy
    states
    rewards
    function VisualizePPHW(k_max)
        k_max+=1
        model = [DataFrame(east=zeros(k_max),north_east=zeros(k_max),north_west=zeros(k_max),west=zeros(k_max),south_west=zeros(k_max),south_east=zeros(k_max)),
        DataFrame(east=zeros(k_max),north_east=zeros(k_max),north_west=zeros(k_max),west=zeros(k_max),south_west=zeros(k_max),south_east=zeros(k_max))]
        policy= [DataFrame(east=zeros(k_max),north_east=zeros(k_max),north_west=zeros(k_max),west=zeros(k_max),south_west=zeros(k_max),south_east=zeros(k_max)),
        DataFrame(east=zeros(k_max),north_east=zeros(k_max),north_west=zeros(k_max),west=zeros(k_max),south_west=zeros(k_max),south_east=zeros(k_max))]
        states=Vector{Tuple{Int64, Int64}}()
        rewards=Vector{Tuple{Int64, Int64}}()

        push!(rewards,(0,0))
        model[1][1,:].=[1/6,1/6,1/6,1/6,1/6,1/6]
        model[2][1,:].=[1/6,1/6,1/6,1/6,1/6,1/6]
        
        return new(model,policy,states,rewards)
    end
end

# const HexWorldRBumpBorder = -1.0 # Reward for falling off hex map
# const HexWorldPIntended = 0.7 # Probability of going intended direction
# const HexWorldDiscountFactor = 0.9
function PredatorPreyHexWorld()
    PredatorPreyHexWorld = PredatorPreyHexWorldMG(
        [
            (-1, 2), (0, 2), (1, 2),
            (-1, 1), (1, 1), (3, 1), (4, 1),
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        ],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        HexWorldDiscountFactor
    )
    return PredatorPreyHexWorld
end




struct MGPolicy
    p # dictionary mapping states to simple game policies
    MGPolicy(p::Base.Generator) = new(Dict(p))
end

# ở ulatr (πi::SimpleGamePolicy)(ai)
(πi::MGPolicy)(s, ai) = πi.p[s](ai)
(πi::SimpleGamePolicy)(s, ai) = πi(ai)

probability(𝒫::MG, s, π, a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))
reward(𝒫::MG, s, π, i) =
    sum(𝒫.R(s, a)[i] * probability(𝒫, s, π, a) for a in joint(𝒫.𝒜))
transition(𝒫::MG, s, π, s′) =
    sum(𝒫.T(s, a, s′) * probability(𝒫, s, π, a) for a in joint(𝒫.𝒜))



mutable struct MGFictitiousPlay
    𝒫 # Markov game
    i # agent index
    Qi # state-action value estimates
    Ni # state-action counts
end
function MGFictitiousPlay(𝒫::MG, i)
    ℐ, 𝒮, 𝒜, R = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R
    # Qi= [(state 12, action 12)  => reward]
    Qi = Dict((s, a) => R(s, a)[i] for s in 𝒮 for a in joint(𝒜))

    # display(Qi)
    # println("----------------")
    # A[j]: [1...6]

    # N[i]= [agent 1, state 12, action chọn 1 trong [1:6]]
    Ni = Dict((j, s, aj) => 1.0 for j in ℐ for s in 𝒮 for aj in 𝒜[j])
    # γ  # discount factor
    # ℐ  # agents
    # 𝒮  # state space
    # 𝒜  # joint action space
    # T  # transition function 
    # R  # joint reward function 
    # (agent, in (12,12), action) => 1
    # display(Ni)
    return MGFictitiousPlay(𝒫, i, Qi, Ni)
end

function (πi::MGFictitiousPlay)(s)
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ


    # SimpleGamePolicy(Dict(1=> count Ni,...6))
    # return Dict(6 keys Ni => xác suất count)
    πi′(i, s) = SimpleGamePolicy(ai => πi.Ni[i, s, ai] for ai in 𝒜[i])

    # MGPolicy(Dict{Tuple{Int64, Int64}, SimpleGamePolicy}
    πi′(i) = MGPolicy(s => πi′(i, s) for s in 𝒮)

    display(i)

    π = [πi′(i) for i in ℐ]
    # display(MGPolicy(s => πi′(i, s) for s in [(1,2),(2,3)]))
    # chưa đọc 
    # display(π[1].p[(1,2)](4))

    # probability(𝒫, s, π, a): xác suất tại s12 chọn a12

    # sum(reward*probability tương ứng)
    # U: đánh giá state s
    U(s, π) = sum(πi.Qi[s, a] * probability(𝒫, s, π, a) for a in joint(𝒜))

    # println(U((1,2),π)) 


    # action value function : Q-funtion: 7.12
    # transition(𝒫, s, π, s′): tổng khả năng transition từ s qua s' với mọi a có thể

    # Q: đánh giá state s trên toàn bộ hex world
    Q(s, π) = reward(𝒫, s, π, i) + γ * sum(transition(𝒫, s, π, s′) * U(s′, π)
                                           for s′ in 𝒮)

    # joint => [SimpleGamePolicy,MGPolicy ]; [MGPolicy,SimpleGamePolicy]
    # SimpleGamePolicy (ai => 1.0), aj  => 0

    # tìm action cho agent i => SimpleGamePolicy => tính Q(ai) => quyến định chọn ai 
    # đối thủ => MGPolicy => tính hết A
    Q(ai) = Q(s, joint(π, SimpleGamePolicy(ai), i))
    # index của max element
    # x=[Q(i) for i in collect(1:6)]
    # println(x)
    # Q là function tính toán giá trị Q-value cho từng elements in [1...6]

    # đi theo action nào thì khả năng nhận dc reward cao nhất
    ai = argmax(Q, 𝒫.𝒜[πi.i])

    display(ai)
    # display(SimpleGamePolicy(ai))

    # return Dict(): action nào có Q-value lớn nhất => xs =1
    return SimpleGamePolicy(ai)
end
function update!(πi::MGFictitiousPlay, s, a, s′,v,iteration)
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ

    # +1: tính U => Q => ai
    for (j, aj) in enumerate(a)
        πi.Ni[j, s, aj] += 1
    end

    # update visualize
    v.policy[i][iteration,a[i]]=1
    
    totalCount = sum(πi.Ni[i,S,ai] for S in 𝒮 for ai in 𝒜[i])
    for ai in 𝒜[i]
        v.model[i][iteration+1,ai]=sum(πi.Ni[i,S,ai] for S in 𝒮)/totalCount
    end
    
    
    

    # action => xác suất count
    πi′(i, s) = SimpleGamePolicy(ai => πi.Ni[i, s, ai] for ai in 𝒜[i])
    # MGPolicy(Dict{Tuple{Int64, Int64}, SimpleGamePolicy}
    πi′(i) = MGPolicy(s => πi′(i, s) for s in 𝒮)

    π = [πi′(i) for i in ℐ]

    U(π, s) = sum(πi.Qi[s, a] * probability(𝒫, s, π, a) for a in joint(𝒜))

    Q(s, a) = R(s, a)[i] + γ * sum(T(s, a, s′) * U(π, s′) for s′ in 𝒮)

    # đánh giá trên toàn bộ actions
    # vd: action 1, state s tới những state khác s sẽ 
    for a in joint(𝒜)
        # Cập nhật lại Q value cho reward của key (s,a) tương ứng
        πi.Qi[s, a] = Q(s, a)
    end
end



function randstep(𝒫::MG, s, a)
    # random dựa trên phân phối chuẩn // trả về state s' [base on propability ở giữa]
    # khả năng xuát hiện cao
    s′ = rand(SetCategorical(𝒫.𝒮, [𝒫.T(s, a, s′) for s′ in 𝒫.𝒮]))
    #display(𝒫.T)
    # display(T(s, a, s′) for s′ in 𝒫.𝒮)
    # display("\n")
    # #display(s′)
    # r là cố định
    r = 𝒫.R(s, a)
    return s′, r
end





function simulate(𝒫::MG, π, k_max, b)
    v = VisualizePPHW(k_max)
    # random vị trí state của 2 agent
    s = rand(b)
    while s[1]==s[2]
        s = rand(b)
    end
    push!(v.states,s)
    # k_max: iteration
    for k = 1:k_max
        # println("s => ", s)
        # (): return key, key la action ai cua SimpleGamePolicy
        # a: (action cua 1, action cua 2)
        a = Tuple(πi(s)() for πi in π)
        
        # println("-----------  a => ", a)
        display(a)
        #random state mới
        s′, r = randstep(𝒫, s, a)
        # println(s," => ",s′)
        for πi in π
            # update lại policy
            update!(πi, s, a, s′,v,k)
        end
        
        # update reward visualize
        
        if(s[1] != s[2])
            if(s[1]==s′[1])
                r[1] = 0
            end
            if(s[2]==s′[2])
                r[2] = 0
            end
        end
        if(k > 1)
            r[1]+=v.rewards[k][1]
            r[2]+=v.rewards[k][2]
        end
        push!(v.rewards,Tuple(r))
        push!(v.states,s′)

        # sử dụng state này làm s
        s = s′

    end

# a dũ helper
    a = Tuple(πi(s)() for πi in π)
    for i in 1:2
        v.policy[i][k_max+1,a[i]]=1
    end

    return v,π
end




p = PredatorPreyHexWorld()
# display(p)
mg = MG(p)
π = [MGFictitiousPlay(mg, i) for i in 1:2]
#display(π)
print("version ----------------------------------------\n\n\n\n\n")
k_max=10
v,policy=simulate(mg, π, k_max, mg.𝒮)

# display(v)
# drawPredatorPreyHW(v.states,v.rewards,k_max)

# visualize

# colName = [:east :north_east :north_west :west :south_west :south_east]
model1= @df v.model[1] plot(0:k_max, [:east :north_east :north_west :west :south_west :south_east],legend=false, xlabel="iteration",title="opponent model - predator")
model2= @df v.model[2] plot(0:k_max, [:east :north_east :north_west :west :south_west :south_east],legend=false,title="opponent model - prey")
policy1= @df v.policy[1] plot(0:k_max,[:east :north_east :north_west :west :south_west :south_east],legend=false,title="policy - predator", framestyle = :none)
policy2= @df v.policy[2] plot(0:k_max, [:east :north_east :north_west :west :south_west :south_east],legend=false,xlabel="iteration",title="policy - prey")


plot(model2,policy1,model1,policy2,layout=(2,2),size=(1000,700), grid=:off)#,ylim=(-0.05,1))
