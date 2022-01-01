include("./discrete_mdp.jl")

# 6 neighborhood of i,j cell in Hex World map
# Example: 1,1 => Neighbor cells are: 2,1 1,2 0,2 0,1 1,0 2,0
function hex_neighbors(hex::Tuple{Int,Int})
    i,j = hex
    [(i+1,j),(i,j+1),(i-1,j+1),(i-1,j),(i,j-1),(i+1,j-1)]
end

#Struct Hex World following MDP
struct HexWorldMDP

    hexes::Vector{Tuple{Int,Int}}     # Hexes is a set of state == HexWorld cells 
    
    mdp::DiscreteMDP    # DiscreteMDP for this problem
    
    special_hex_rewards::Dict{Tuple{Int,Int}, Float64} # Rewards: cell <-> reward value 
    
    #Constructor HexWorldMDP
    function HexWorldMDP(
        hexes::Vector{Tuple{Int,Int}},
        r_bump_border::Float64,
        p_intended::Float64,
        special_hex_rewards::Dict{Tuple{Int,Int}, Float64},
        γ::Float64,)

        nS = length(hexes) + 1 # plus one goal state
        nA = 6 # 6 directions per cell 
        s_absorbing = nS #goal state setup

        #Constructor T, R 
        T = zeros(Float64, nS, nA, nS)
        R = zeros(Float64, nS, nA)

        p_veer = (1.0 - p_intended)/2 #probability of veering left or right

        # traverse a tile map
        for s in 1 : length(hexes)
            hex = hexes[s]
            if !haskey(special_hex_rewards, hex)
                # Rewardless cell
                # Action taken from a normal tile
                neighbors = hex_neighbors(hex)
                for (a,neigh) in enumerate(neighbors)
                    
                    # Intended transition: 0.7
                    s′ = findfirst(h -> h == neigh, hexes)
                    if s′ == nothing
                        # This state has no neighborcell
                        s′ = s
                        R[s,a] += r_bump_border*p_intended
                    end
                    T[s,a,s′] += p_intended #setup Intended transition: 0.7

                    # Unintended veer left: 0.15
                    a_left = mod1(a+1, nA) 
                    neigh_left = neighbors[a_left]
                    s′ = findfirst(h -> h == neigh_left, hexes) #s action2 to s'
                    if s′ == nothing
                        # This state has no left neighborcell
                        s′ = s
                        R[s,a] += r_bump_border*p_veer
                    end
                    T[s,a,s′] += p_veer #setup Unintended veer left: 0.15

                    # Unintended veer right: 0.15
                    a_right = mod1(a-1, nA) #action6
                    neigh_right = neighbors[a_right]
                    s′ = findfirst(h -> h == neigh_right, hexes)
                    if s′ == nothing
                        # This state has no right neighborcell
                        s′ = s
                        R[s,a] += r_bump_border*p_veer
                    end
                    T[s,a,s′] += p_veer #setup Unintended veer right: 0.15
                end
            else
                # Goal/ termial state setup
                # Action taken from an absorbing hex
                # In absorbing hex, the action automatically takes agent to the absorbing state 
                # Agent will get the reward.
                for a in 1 : nA
                    T[s,a,s_absorbing] = 1.0
                    R[s,a] += special_hex_rewards[hex]
                end
            end
        end

        # Absorbing state stays where it is and gets no reward.
        for a in 1 : nA
            T[s_absorbing,a,s_absorbing] = 1.0
        end

        mdp = DiscreteMDP(T,R,γ) #Constructor Discrete MDP

        return new(hexes, mdp, special_hex_rewards) #HexWorldMDP
    end
end

const HexWorldRBumpBorder = -1.0 # Reward for falling off hex map
const HexWorldPIntended = 0.7 # Probability of going intended direction
const HexWorldDiscountFactor = 0.9 #Discount factor

function HexWorld()
    HexWorld = HexWorldMDP(
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(-1,2),
         (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
         (8,2),(4,1),(5,0),(6,0),(7,0),(7,1),(8,1),(9,0)], #states/cells
        HexWorldRBumpBorder,
        HexWorldPIntended,
        Dict{Tuple{Int,Int}, Float64}(
            (0,1)=>  5.0, 
            (2,0)=>-10.0,
            (9,0)=> 10.0, 
        ),
        HexWorldDiscountFactor
    )
    return HexWorld
end

#HexWorldMDP to MDP
function MDP(mdp::HexWorldMDP)
    return MDP(mdp.mdp.T,mdp.mdp.R,mdp.mdp.γ)
end
