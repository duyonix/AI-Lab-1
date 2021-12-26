using Luxor, Colors

function drawHexGrid()
    Drawing(1000, 200)
    radius = 30
    grid = GridHex(Point(50,50), radius, 550)
    arrow(O, Point(O.x + (sqrt(3) * radius)/2, 0))
    background("white")
    for i in 1:30
        sethue("gray")
        p = nextgridpoint(grid)
        ngon(p, radius-5, 6, pi/2, :fillstroke)
        sethue("white")
        text(string(i), p, halign=:center)
    end
    finish()
    preview()
end

drawHexGrid()

function drawHexWorld(numCells, negaPosition, posPosition, rewards)
    Drawing(1000, 200)
    radius = 30
    grid = GridHex(Point(50,50), radius, 550)
    arrow(O, Point(O.x + (sqrt(3) * radius)/2, 0))
    background("white")
    for i in 1:numCells
        sethue("gray")
        if i == negaPosition
            sethue("red")
        end
        if i == posPosition
            sethue("blue")
        end
        p = nextgridpoint(grid)
        ngon(p, radius-5, 6, pi/2, :fillstroke)
        sethue("white")
        text(string(rewards[i]), p, halign=:center)
    end
    finish()
    preview()
end

numCells = 30
rewards = rand((-10,0,5,10), numCells)
println(rewards)
redRe = 1
blueRe = 30
drawHexWorld(numCells, redRe, blueRe, rewards)