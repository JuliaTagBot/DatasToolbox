using DatasToolbox
using DataFrames

df = DataFrame(rand(100, 4))

dh = DataHandler{Float64}(df, input_cols=[:x1, :x2], output_cols=[:x3],
                          testfrac=0.1, shuffle=true)

expr = macroexpand(:(@split dh x1 .> 0.8))
@split(dh, x1 .> alpha)
