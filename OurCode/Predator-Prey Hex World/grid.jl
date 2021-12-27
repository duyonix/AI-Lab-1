using Luxor, Colors

function drawArrow(centerHexagon, startState, endState, isCaptured)
    # predator
    sethue("red")
    circle(centerHexagon[startState[1]], 10, :fill)
    if (startState[1] != endState[1])
        arrow(centerHexagon[startState[1]], centerHexagon[endState[1]], linewidth = 6, arrowheadlength = 20, arrowheadangle = pi / 6)
    end


    # prey
    sethue("cyan")
    circle(centerHexagon[startState[2]], 6, :fill)
    
    if(isCaptured)
        sethue("grey")
    end
    if (startState[2] != endState[2])
        arrow(centerHexagon[startState[2]], centerHexagon[endState[2]], linewidth = 4, arrowheadlength = 15, arrowheadangle = pi / 6)
    end
end

function drawPredatorPreyHW(cacheStates,rewards, iterations)
    discrete = [2, 3, 4, 8, 10, 12, 13, 15, 16, 17, 18, 19]
    state = 1:12

    radius = 40
    height = 220
    width = 500
    # Drawing(width*2, height*iterations/2,"predator-prey.png")
    Drawing(width * 2, height * floor(iterations / 2))
    background("white")
    p=nothing
    for iter in 1:iterations
        centerHexagon = Vector{Point}()
        startPoint = nothing

        if (iter % 2 == 0)
            startPoint = Point(50 + width, 50 + height * (floor(iter / 2) - 1))
            grid = GridHex(startPoint, radius, 500 + width)
        else
            startPoint = Point(50, 50 + height * (floor(iter / 2)))
            grid = GridHex(startPoint, radius, 500)
        end
        sethue("black")
        fontsize(25)
        text(string(iter-1), startPoint,halign=:center, valign = :middle)
        fontsize(20)

        sethue("red")
        text(string(rewards[iter][1]),startPoint+Point(350,-10),halign=:right, valign = :bottom)
        
        sethue("blue")
        text(string(rewards[iter][2]),startPoint+Point(400,-10),halign=:right, valign = :bottom)

        j=1
        sethue("white")
        fontsize(17)
        for i in 1:20
            if i in discrete
                sethue(0, 0, 0)
                p = nextgridpoint(grid)
                push!(centerHexagon, p)
                ngon(p, radius - 5, 6, pi / 2, :fillstroke)
                sethue("white")

                text(string(state[j]), p - Point(0, 20), halign = :center, valign = :middle)
                j += 1

            else
                sethue("white")
                p = nextgridpoint(grid)
            end

        end
        isCaptured = false
        if(iter > 0 && rewards[iter+1][1]==10)
            isCaptured = true
        end
        drawArrow(centerHexagon,cacheStates[iter],cacheStates[iter+1],isCaptured)
    end

    finish()
    preview()
end

