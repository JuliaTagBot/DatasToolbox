using DatasToolbox

const N = 20

srand(999)

df = DataFrame(A=rand(1:3, N), B=rand(N), C=rand(N))

gdh = GroupedDataHandler{Float64}(df, [:A], input_cols=[:B], output_cols=[:C])

split!(gdh, round(Int64, N/2), assign=false)

assign!(gdh)

X_train, y_train = getTrainData(gdh)

X_test, y_test = getTestData(gdh)

ŷ = Dict((1,)=>[1.0 for i ∈ 1:length(X_test[(1,)])],
         (2,)=>[2.0 for i ∈ 1:length(X_test[(2,)])],
         (3,)=>[3.0 for i ∈ 1:length(X_test[(3,)])])

odf = getTestAnalysisData(gdh, ŷ)

