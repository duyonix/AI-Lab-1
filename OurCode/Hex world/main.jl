

using Test

@testset "hexworld.jl" begin
    m = HexWorld()
    hexes = m.hexes
    @test p.n_states(m) == length(hexes) + 1 && p.ordered_states(m) == 1:length(hexes) + 1
    @test p.n_actions(m) == 6 && p.ordered_actions(m) == 1:6
    @test p.discount(m) == 0.9
    @test p.state_index(m, 1) == 1
    state = rand(1:p.n_states(m))
    action = rand(1:p.n_actions(m))
    p.transition(m, state, action)
    state_ = p.generate_s(m, state, action)
    reward = p.reward(m, state, action)
    @test state_ in p.ordered_states(m)
    @test reward <= 10
    @test p.generate_sr(m, state, action)[1] in p.ordered_states(m) && p.generate_sr(m, state, action)[2] <= 10
    @test p.generate_start_state(m) in p.ordered_states(m)
    @test p.hex_distance(rand(hexes), rand(hexes)) >= 0
    mdp = MDP(m)
end