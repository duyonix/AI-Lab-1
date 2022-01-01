using StatsPlots

function visualizeRPS(v)
    model1 = @df v.model[1] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], xlabel = "iteration", title = "opponent model (agent 1)", ylim = (-0.05, 1))
    model2 = @df v.model[2] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], title = "opponent model (agent 2)", ylim = (-0.05, 1))
    policy1 = @df v.policy[1] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], legend = false, title = "policy agent 1", ylim = (-0.05, 1))
    policy2 = @df v.policy[2] plot(1:k_max, [:rock :paper :scissors], colour = [:red :blue :green], legend = false, xlabel = "iteration", title = "policy agent 2", ylim = (-0.05, 1))

    r1 = [reward[1] for reward in v.rewards]
    r2 = [reward[2] for reward in v.rewards]

    reward1 = plot(1:k_max, r1, legend = false, title = "reward agent 1")
    reward2 = plot(1:k_max, r2, legend = false, title = "reward agent 2", xlabel = "iteration")

    plot(model2, policy1, reward1, model1, policy2, reward2, layout = (2, 3), size = (1200, 750), grid = :off)
end