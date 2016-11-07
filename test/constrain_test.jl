using DataFrames
using DatasToolbox

df = DataFrame(A=1:2:20, B=5:14)

const M = 5

dict = Dict()
expr = :((:A .< 10) && (:B % 2 == 0))
DatasToolbox._checkConstraintExpr!(expr, dict)


mac = macroexpand(:(@constrain(df, (:A .< 10) && (:B % 2 == 0))))

cdf = @constrain(df, (:A .< 10) && (:B % 2 == 0))



