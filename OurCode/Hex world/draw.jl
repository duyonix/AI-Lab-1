# using Hexagons
using Test
x=hexagon(1,-1,0)
# print(typeof(x))

neighbors(x)

convert(HexagonOffsetOddR, HexagonAxial(2, 4))

x, y = center(HexagonAxial(2, 3))
@test isapprox(x,7.06217782649107)
@test isapprox(y,5.5)

h = cube_round(23.5, 4.67)
display=collect(vertices(HexagonAxial(-2, -3)))
function arrayonplot()
    for i =1:6
        plot(display[i])
    end
end  
using Plots