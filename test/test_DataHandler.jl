using DatasToolbox
using DataFrames

df = DataFrame(100*rand(100, 4))

dh = DataHandler{Float64}(df, input_cols=[:x1, :x2], output_cols=[:x3],
                          normalize_cols=[:x1, :x2],
                          testfrac=0.1, shuffle=true, userange=true)

function test1!(dh)
    @split! dh (x1 .≥ 50.0) | (x2 .< 20.0)
end

test1!(dh)

