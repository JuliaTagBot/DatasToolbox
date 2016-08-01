using DatasToolbox
using DataFrames
# using ScikitLearn
# fit! = ScikitLearn.fit!
# predict = ScikitLearn.predict
# @sk_import ensemble: RandomForestRegressor
using XGBoost
predict = XGBoost.predict

df_train = DataFrame(ξ=1:2:1000, ζ=2:2:1000, τ=1:500)
df_test = DataFrame(ξ=502:2:520, ζ=501:2:520, τ=501:510)
df = [df_train; df_test]

tsh = TimeSeriesHandler{Float64}(df, :τ, 5, input_cols=[:ξ], output_cols=[:ξ],
                                 n_test_sequences=0,
                                 userange=true)
split!(tsh, 505)
# calling this again to be safe
assign!(tsh)

X_train, y_train = getSquashedTrainData(tsh, flatten=true)

booster = xgboost(X_train, 500, label=y_train, objective="reg:linear")

# yhat = generateTest(X -> predict(booster, X), tsh, on_matrix=true)
yhat = generateSequence(X -> predict(booster, X), tsh, 10, on_matrix=true)


