# DatasToolbox

___NOTE:___ DatasToolbox is being split into multiple packages.  For now I'm keeping it around
as an alias for importing all said packages.

This is a set of tools for performing common, generic manipulations on data, primarily for 
machine learning and optimization.  It also contains a set of tools for easily converting
from Python in a type-correct way.

Note that there is currently a bug in the documentation generator that causes it to also
list private methods.  All such methods have names that start with `_`.

## Converting From Python Objects
There exist Python APIs for accessing data stored in Apache parquet or avro as well as SQL
databases, so it is common to have to convert from Python (particularly pandas) objects.
Python has some extremely undesirable properties in regard to data manipulation.  In 
particular, its tendency to hide types can be very cumbersome.  Therefore, some significant
effort has been put into making sure that conversions are robust, and usually give 
type-correct Julia objects.

The must commonly used tool is expected to be `convertPyDF(pydf)`.  This expression will
return ad Julia dataframe from the pandas dataframe `pydf`.  For the most part, the result
should have the appropriate types, though in some cases there may be columns which have 
eltype `Any` even when inappropriate. See the API docs for the various helper functions
which are used to facilitate the conversion.

One can also convert a pickled Python dataframe stored on local disk to a Julia dataframe
using, for example `loadPickledDF("filename.pkl")`.


## DataHandler
The main functionality currently available in `DatasToolbox` is supplied with the 
`DataHandler` type.  This type is designed for converting dataframes into usable machine
learning data in the form of input, target pairs ``X, y``.  The design philosophy is that
hyper-parameter tuning and testing should be done completely separately.

### DataHandler Example
As a somewhat silly example, suppose we have a dataframe which stores (experimentally 
determined) particle properties, and we would like to classify the particle as a quark, 
lepton, scalar, or gauge boson and that we are doing this with gradient boosted trees.
Suppose the dataframe contains 4-momenta, estimated spin and electric charge of particles,
all as floats.

The columns which we want to use as input for our classification can be, for example,
```julia
input_cols = [:E, :px, :py, :pz, :S, :Q]
```
(yes, this example is ridiculous for a number of reasons, but its just to demonstrate
how things work).  From this we want to predict the type of particle so let's declare
```julia
output_cols = [:ptype]
```

Now we construct the `DataHandler` object.
```julia
dh = DataHandler{Float64}(df, input_cols=input_cols, output_cols=output_cols)
```
The type parameter is the type which the train and test data will ultimately be converted
to.  For now it is assumed that these are all the same (this conversion to all the same
type turns out to be useful for many machine learning methods).

Next, we can randomly split the dataframe into training and test (validation) sets.  
```julia
split!(dh, 0.2)
```
This assigns one fifth of the data to the test set, and the rest to the training set.
In realistic scenarios, one more often has to do splits like
```julia
split!(dh, df[:Q] .> 0.0)
```
(or something less ridiculous, but you get the idea).  Once we have split the data we are 
ready to extract it in useful form.  First call 
```julia
assign!(dh)
```
to convert the data to properly formatted arrays.  Then, to retrieve them do
```julia
X_train, y_train = getTrainData(dh, flatten=true)
```
The flatten argument ensures that `y_train` is a rank-1 array as opposed to a rank-2 array
with 1 column.

At this point you can train your classifier however you normally would.  In this example
we'll use the gradient boosted tree library `xgboost`
```julia
boost = xgboost(X_train, N_ITERATIONS, label=y_train, eta=η)
```
Then we can get the test data and perform a test
```julia
X_test, y_test = getTestData(dh, flatten=true)

ŷ = predict(boost, X_test)
```

There are also tools for analyzing the test data.  To create a useful dataframe for testing
one can do
```julia
output = getTestAnalysisData(dh, ŷ, names=[:ptype_predicted])
```
The names argument specifies the name of the prediction column in the resulting dataframe.
Alternatively one can just use `y_test` to create whatever test statistics or plots one wants.


### TimeSeriesHandler Example
A far more complicated data manipulation task is preparing time series data.  For this
we supply the type `TimeSeriesHandler`.  Suppose we have a dataframe containing the columns
`:y` and `:τ` where `:τ` is a time index.  To declare the handler
```julia
tsh = TimeSeriesHandler{Float64}(df, :τ, SEQ_LENGTH, input_cols=[:y], output_cols=[:y],
                                 normalize_cols=[:y])
```
Note that here the input and output columns are the same, because we are auto-regressing
``y`` on itself, but the input and output columns can be whatever you want.  In this case
we also want to center and rescale the data so that it is more appropriate as input
to a recurrent neural network.  This option is also avaialbe for the `DataHandler` object
and works the same way as it will in this example.  It is required that the input dataframe
has a time index.  `SEQ_LENGTH` provides the length of sequences in the training and test
data.

Now, one can create a train-test split and generate the properly formatted data.
```julia
split!(tsh, τ_split)
computeNormalizeParamters!(tsh)
normalize!(tsh)
assign!(tsh)

X_train, y_train = getTrainData(tsh)
```
The function `computeNormalizeParameters!` computes the parameters that are necessary for
performing the centering and rescaling.  After `normalize!` is called the returned data
will be properly normalized.  In this example `X_train` is a rank-3 tensor appropriate for
input to recurrent neural networks.  To instead return a matrix where each row is of length
`SEQ_LENGTH` one can call `getSquashedTrainData`.

`X_train, y_train` are properly formatted arrays that can be fed into the training function
of whatever method you are using.

When testing a time series regression, it is often desirable to create a sequence of a 
specified length by predicting on the previous ``N`` points.  This requires extremely
complicated data manipulation, but can be done with
```julia
ŷ = generateSequence(predict, tsh, PREDICTION_LENGTH)
```
The first argument should be the function that is used to make predictions.  This will
generate a sequence by predicting on the last ``N`` points of the training set, then 
predicting on the last ``N-1`` points of the training set and the 1 point which was just
predicted, then the last ``N-2`` points of the training set and the 2 points which were
just predicted and so forth.  If the predicted sequence is of the same length as the test
set one can still do
```julia
output = getTestAnalysisData(tsh, ŷ, names=[:ŷ])
```



## API Docs

```@autodocs
Modules = [DatasToolbox]
Private = false
```
