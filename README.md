
<a id='DatasToolbox-1'></a>

# DatasToolbox


___NOTE:___ DatasToolbox is being split into multiple packages.  For now I'm keeping it around as an alias for importing all said packages.


This is a set of tools for performing common, generic manipulations on data, primarily for  machine learning and optimization.  It also contains a set of tools for easily converting from Python in a type-correct way.


Note that there is currently a bug in the documentation generator that causes it to also list private methods.  All such methods have names that start with `_`.


<a id='Converting-From-Python-Objects-1'></a>

## Converting From Python Objects


There exist Python APIs for accessing data stored in Apache parquet or avro as well as SQL databases, so it is common to have to convert from Python (particularly pandas) objects. Python has some extremely undesirable properties in regard to data manipulation.  In  particular, its tendency to hide types can be very cumbersome.  Therefore, some significant effort has been put into making sure that conversions are robust, and usually give  type-correct Julia objects.


The must commonly used tool is expected to be `convertPyDF(pydf)`.  This expression will return ad Julia dataframe from the pandas dataframe `pydf`.  For the most part, the result should have the appropriate types, though in some cases there may be columns which have  eltype `Any` even when inappropriate. See the API docs for the various helper functions which are used to facilitate the conversion.


One can also convert a pickled Python dataframe stored on local disk to a Julia dataframe using, for example `loadPickledDF("filename.pkl")`.


<a id='DataHandler-1'></a>

## DataHandler


The main functionality currently available in `DatasToolbox` is supplied with the  `DataHandler` type.  This type is designed for converting dataframes into usable machine learning data in the form of input, target pairs $X, y$.  The design philosophy is that hyper-parameter tuning and testing should be done completely separately.


<a id='DataHandler-Example-1'></a>

### DataHandler Example


As a somewhat silly example, suppose we have a dataframe which stores (experimentally  determined) particle properties, and we would like to classify the particle as a quark,  lepton, scalar, or gauge boson and that we are doing this with gradient boosted trees. Suppose the dataframe contains 4-momenta, estimated spin and electric charge of particles, all as floats.


The columns which we want to use as input for our classification can be, for example,


```julia
input_cols = [:E, :px, :py, :pz, :S, :Q]
```


(yes, this example is ridiculous for a number of reasons, but its just to demonstrate how things work).  From this we want to predict the type of particle so let's declare


```julia
output_cols = [:ptype]
```


Now we construct the `DataHandler` object.


```julia
dh = DataHandler{Float64}(df, input_cols=input_cols, output_cols=output_cols)
```


The type parameter is the type which the train and test data will ultimately be converted to.  For now it is assumed that these are all the same (this conversion to all the same type turns out to be useful for many machine learning methods).


Next, we can randomly split the dataframe into training and test (validation) sets.  


```julia
split!(dh, 0.2)
```


This assigns one fifth of the data to the test set, and the rest to the training set. In realistic scenarios, one more often has to do splits like


```julia
split!(dh, df[:Q] .> 0.0)
```


(or something less ridiculous, but you get the idea).  Once we have split the data we are  ready to extract it in useful form.  First call 


```julia
assign!(dh)
```


to convert the data to properly formatted arrays.  Then, to retrieve them do


```julia
X_train, y_train = getTrainData(dh, flatten=true)
```


The flatten argument ensures that `y_train` is a rank-1 array as opposed to a rank-2 array with 1 column.


At this point you can train your classifier however you normally would.  In this example we'll use the gradient boosted tree library `xgboost`


```julia
boost = xgboost(X_train, N_ITERATIONS, label=y_train, eta=η)
```


Then we can get the test data and perform a test


```julia
X_test, y_test = getTestData(dh, flatten=true)

ŷ = predict(boost, X_test)
```


There are also tools for analyzing the test data.  To create a useful dataframe for testing one can do


```julia
output = getTestAnalysisData(dh, ŷ, names=[:ptype_predicted])
```


The names argument specifies the name of the prediction column in the resulting dataframe. Alternatively one can just use `y_test` to create whatever test statistics or plots one wants.


<a id='TimeSeriesHandler-Example-1'></a>

### TimeSeriesHandler Example


A far more complicated data manipulation task is preparing time series data.  For this we supply the type `TimeSeriesHandler`.  Suppose we have a dataframe containing the columns `:y` and `:τ` where `:τ` is a time index.  To declare the handler


```julia
tsh = TimeSeriesHandler{Float64}(df, :τ, SEQ_LENGTH, input_cols=[:y], output_cols=[:y],
                                 normalize_cols=[:y])
```


Note that here the input and output columns are the same, because we are auto-regressing $y$ on itself, but the input and output columns can be whatever you want.  In this case we also want to center and rescale the data so that it is more appropriate as input to a recurrent neural network.  This option is also avaialbe for the `DataHandler` object and works the same way as it will in this example.  It is required that the input dataframe has a time index.  `SEQ_LENGTH` provides the length of sequences in the training and test data.


Now, one can create a train-test split and generate the properly formatted data.


```julia
split!(tsh, τ_split)
computeNormalizeParamters!(tsh)
normalize!(tsh)
assign!(tsh)

X_train, y_train = getTrainData(tsh)
```


The function `computeNormalizeParameters!` computes the parameters that are necessary for performing the centering and rescaling.  After `normalize!` is called the returned data will be properly normalized.  In this example `X_train` is a rank-3 tensor appropriate for input to recurrent neural networks.  To instead return a matrix where each row is of length `SEQ_LENGTH` one can call `getSquashedTrainData`.


`X_train, y_train` are properly formatted arrays that can be fed into the training function of whatever method you are using.


When testing a time series regression, it is often desirable to create a sequence of a  specified length by predicting on the previous $N$ points.  This requires extremely complicated data manipulation, but can be done with


```julia
ŷ = generateSequence(predict, tsh, PREDICTION_LENGTH)
```


The first argument should be the function that is used to make predictions.  This will generate a sequence by predicting on the last $N$ points of the training set, then  predicting on the last $N-1$ points of the training set and the 1 point which was just predicted, then the last $N-2$ points of the training set and the 2 points which were just predicted and so forth.  If the predicted sequence is of the same length as the test set one can still do


```julia
output = getTestAnalysisData(tsh, ŷ, names=[:ŷ])
```


<a id='API-Docs-1'></a>

## API Docs

<a id='Base.Dict-Tuple{Array{K,1},Array{V,1}}' href='#Base.Dict-Tuple{Array{K,1},Array{V,1}}'>#</a>
**`Base.Dict`** &mdash; *Method*.



**DatasToolbox**

`DatasToolbox` provides the following new constructors for `Dict`:

```
Dict(keys, values)
Dict(df, keycol, valcol)
```

One can provide `Dict` with (equal length) vector arguments.  The first vector provides a list of keys, while the second provides a list of values. If the vectors are `NullableVector`, only key, value pairs with *both* their elements non-null will be added.

<a id='DatasToolbox.AbstractDH' href='#DatasToolbox.AbstractDH'>#</a>
**`DatasToolbox.AbstractDH`** &mdash; *Type*.



```
AbstractDH{T}
```

Abstract base class for data handler objects.

<a id='DatasToolbox.DataHandler' href='#DatasToolbox.DataHandler'>#</a>
**`DatasToolbox.DataHandler`** &mdash; *Type*.



```
DataHandler{T} <: AbstractDH{T}
```

Type for handling datasets.  This is basically a wrapper for a dataframe with methods for splitting it into training and test sets and creating input and output numerical arrays.  It is intended that most reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.

<a id='DatasToolbox.TimeSeriesHandler' href='#DatasToolbox.TimeSeriesHandler'>#</a>
**`DatasToolbox.TimeSeriesHandler`** &mdash; *Type*.



```
TimeSeriesHandler{T} <: AbstractDH{T}
```

Type for handling time series data.  As with DataHandler it is intended taht most of the reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.

<a id='Base.LinAlg.normalize!-Tuple{DatasToolbox.AbstractDH{T}}' href='#Base.LinAlg.normalize!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`Base.LinAlg.normalize!`** &mdash; *Method*.



```
normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
normalizeTrain!(dh::AbstractDH)
normalizeTest!(dh::AbstractDH)
```

Centers and rescales the columns set by `normalize_cols` in the `DataHandler` constructor.

<a id='Base.Random.shuffle!-Tuple{DataTables.DataTable}' href='#Base.Random.shuffle!-Tuple{DataTables.DataTable}'>#</a>
**`Base.Random.shuffle!`** &mdash; *Method*.



```
shuffle!(df::DataTable)
```

Shuffles a dataframe in place.

<a id='Base.Random.shuffle!-Tuple{DatasToolbox.AbstractDH}' href='#Base.Random.shuffle!-Tuple{DatasToolbox.AbstractDH}'>#</a>
**`Base.Random.shuffle!`** &mdash; *Method*.



```
shuffle!(dh::AbstractDH)
```

Shuffles the main dataframe of the DataHandler.

<a id='Base.Serializer.deserialize-Tuple{AbstractString}' href='#Base.Serializer.deserialize-Tuple{AbstractString}'>#</a>
**`Base.Serializer.deserialize`** &mdash; *Method*.



```
deserialize(filename::AbstractString)
```

Opens a file from the local file system and deserializes what it finds there. This is quite similar to the functionality in `Base` except that the default  `deserialize` method requires an `IOStream` object instead of a file name so this eliminates an extra line of code.

<a id='Base.Serializer.serialize-Tuple{AbstractString,Any}' href='#Base.Serializer.serialize-Tuple{AbstractString,Any}'>#</a>
**`Base.Serializer.serialize`** &mdash; *Method*.



```
serialize(filename::AbstractString, object)
```

Serializes an object and stores on the local file system to file `filename`. This is quite similar to the functionality in `Base`, except that the default `serialize` method requires an `IOStream` object instead of a file name, so this eliminates an extra line of code.

<a id='Base.convert-Tuple{Union{Type{Int32},Type{Int64}},NullableArrays.NullableArray{Float32,N}}' href='#Base.convert-Tuple{Union{Type{Int32},Type{Int64}},NullableArrays.NullableArray{Float32,N}}'>#</a>
**`Base.convert`** &mdash; *Method*.



```
convert(dtype, a)
```

This converts a column of floats that should have been ints, but got converted to floats because it has missing values which were converted to NaN's. The supplied `NullableArray` should have eltype `Float32` or `Float64`.

<a id='DatasToolbox.applyCatConstraints-Tuple{Dict,DataTables.DataTable}' href='#DatasToolbox.applyCatConstraints-Tuple{Dict,DataTables.DataTable}'>#</a>
**`DatasToolbox.applyCatConstraints`** &mdash; *Method*.



```
applyCatConstraints(dict, df[, kwargs])
```

Returns a copy of the dataframe `df` with categorical constraints applied.  `dict` should  be a dictionary with keys equal to column names in `df` and values equal to the categorical values that column is allowed to take on.  For example, to select gauge bosons we can pass `Dict(:PID=>[i for i in 21:24; -24])`.  Alternatively, the values in the dictionary can be functions which return boolean values, in which case the returned dataframe will be the one with column values for which the functions return true.

Note that this requires that the dictionary values are either `Vector` or `Function`  (though one can of course mix the two types).

Alternatively, instead of passing a `Dict` one can pass keywords, for example `applyCatConstraints(df, PID=[i for i in 21:24; -24])`.

<a id='DatasToolbox.assign!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.assign!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.assign!`** &mdash; *Method*.



```
assign!(dh::AbstractDH)
```

Assigns training and test data in the data handler.

<a id='DatasToolbox.assign!-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.assign!-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.assign!`** &mdash; *Method*.



```
assign!(dh::TimeSeriesHandler; sort::Bool=true)
```

Assigns both training and testing data for the `TimeSeriesHandler`.

<a id='DatasToolbox.assignTest!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.assignTest!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.assignTest!`** &mdash; *Method*.



```
assignTest!(dh::AbstractDH)
```

Assigns the test data in the data handler.

<a id='DatasToolbox.assignTest!-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.assignTest!-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.assignTest!`** &mdash; *Method*.



```
assignTest!(dh[, df; sort=true])
```

Assigns the test data.  X output will be of shape (samples, seq_length, seq_width). One should be extremely careful if not sorting.

If a dataframe is provided, data will be assigned from it.

Note that in the time series case this isn't very useful.  One should instead use one of the assigned prediction functions.

<a id='DatasToolbox.assignTrain!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.assignTrain!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh::AbstractDH)
```

Assigns the training data in the data handler so it can be retrieved in proper form.

<a id='DatasToolbox.assignTrain!-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.assignTrain!-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh[, df; sort=true, parallel=false])
```

Assigns the training data.  X output will be of shape (samples, seq_length, seq_width). If `sort` is true, will sort the dataframe first.  One should be extremely careful if `sort` is false.  If `parallel` is true the data will be generated in parallel (using  workers, not threads).  This is useful because this data manipulation is complicated and potentially slow.

If a dataframe is provided, data will be assigned from it.  Alternatively, one can provide a vector of dataframes.  This is useful because sequences which cross the boundaries of the dataframes will *not* be created.

**TODO** I'm pretty sure the parallel version isn't working right because it doesn't use shared arrays.  Revisit in v0.5 with threads.

<a id='DatasToolbox.computeNormalizeParameters!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.computeNormalizeParameters!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.computeNormalizeParameters!`** &mdash; *Method*.



```
computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
```

Gets the parameters for centering and rescaling from either the training dataset  (`dataset=:dfTrain`) or the test dataset (`dataset=:dfTest`).

Does this using the training dataframe by default, but can be set to use test. Exits normally if this doesn't need to be done for any columns.

This should always be called before `normalize!`, that way you have control over what dataset the parameters are computed from.

<a id='DatasToolbox.constrain-Tuple{DataTables.AbstractDataTable,Dict{K<:Symbol,V<:Function}}' href='#DatasToolbox.constrain-Tuple{DataTables.AbstractDataTable,Dict{K<:Symbol,V<:Function}}'>#</a>
**`DatasToolbox.constrain`** &mdash; *Method*.



```
constrain(df, dict)
constrain(df, kwargs...)
constrain(df, cols, func)
```

Returns a subset of the dataframe `df` for which the column `key` satisfies  `value(df[i, key]) == true`.  Where `(key, value)` are the pairs in `dict`.   Alternatively one can use keyword arguments instead of a `Dict`.

Also, one can pass a function the arguments of which are elements of columns specified by `cols`.

<a id='DatasToolbox.convertArray-Tuple{Type{DataArrays.DataArray},NullableArrays.NullableArray{T,N}}' href='#DatasToolbox.convertArray-Tuple{Type{DataArrays.DataArray},NullableArrays.NullableArray{T,N}}'>#</a>
**`DatasToolbox.convertArray`** &mdash; *Method*.



```
convertArray(array_type, a)
```

Convert between `NullableArray`s and `DataArray`s.  The former is used by `DataTables` while the latter is used by `DataFrames`.

Note that this is named `convertArray` rather than simply `convert` so as not to conflict with the existing definitions for `convert`.

<a id='DatasToolbox.convertNulls!-Tuple{Array{T<:AbstractFloat,1},T<:AbstractFloat}' href='#DatasToolbox.convertNulls!-Tuple{Array{T<:AbstractFloat,1},T<:AbstractFloat}'>#</a>
**`DatasToolbox.convertNulls!`** &mdash; *Method*.



```
convertNulls!{T}(A::Array{T, 1}, newvalue::T)
```

Converts all null values (NaN's and Nullable()) to a particular value. Note this has to check whether the type is Nullable.

<a id='DatasToolbox.convertNulls!-Tuple{DataTables.DataTable,Array{Symbol,1},Any}' href='#DatasToolbox.convertNulls!-Tuple{DataTables.DataTable,Array{Symbol,1},Any}'>#</a>
**`DatasToolbox.convertNulls!`** &mdash; *Method*.



```
convertNulls!(df::DataTable, cols::Vector{Symbol}, newvalue::Any)
```

Convert all null values in columns of a DataTable to a particular value.

There is also a method for passing a single column symbol, not as a vector.

<a id='DatasToolbox.convertNulls-Tuple{NullableArrays.NullableArray{T,N},T}' href='#DatasToolbox.convertNulls-Tuple{NullableArrays.NullableArray{T,N},T}'>#</a>
**`DatasToolbox.convertNulls`** &mdash; *Method*.



```
convertNulls{T}(A, newvalue)
```

Converts all null vlaues (NaN's and Nullable()) to a particular value. This is a wrapper added for sake of naming consistency.

<a id='DatasToolbox.convertPyColumn-Tuple{PyCall.PyObject}' href='#DatasToolbox.convertPyColumn-Tuple{PyCall.PyObject}'>#</a>
**`DatasToolbox.convertPyColumn`** &mdash; *Method*.



```
convertPyColumn(pycol::PyObject)
```

Converts a column of a pandas array to a Julia `NullableArray`.

<a id='DatasToolbox.convertPyDF-Tuple{PyCall.PyObject}' href='#DatasToolbox.convertPyDF-Tuple{PyCall.PyObject}'>#</a>
**`DatasToolbox.convertPyDF`** &mdash; *Method*.



```
convertPyDF(pydf[, fixtypes=true])
```

Converts a pandas dataframe to a Julia one.  

Note that it is difficult to infer the correct types of columns which contain references to Python objects.  If `fixtypes`, this will attempt to convert any column with eltype `Any` to the proper type.

<a id='DatasToolbox.convertWeakRefStrings-Tuple{DataTables.AbstractDataTable}' href='#DatasToolbox.convertWeakRefStrings-Tuple{DataTables.AbstractDataTable}'>#</a>
**`DatasToolbox.convertWeakRefStrings`** &mdash; *Method*.



```
convertWeakRefStrings(df)
convertWeakRefStrings!(df)
```

Converts all columns with eltype `Nullable{WeakRefString}` to have eltype `Nullable{String}`. `WeakRefString` is a special type of string used by the feather package to improve deserialization performance.

Note that this will no longer be necessary in Julia 0.6.

<a id='DatasToolbox.copyColumns-Tuple{DataTables.DataTable}' href='#DatasToolbox.copyColumns-Tuple{DataTables.DataTable}'>#</a>
**`DatasToolbox.copyColumns`** &mdash; *Method*.



```
copyColumns(df::DataTable)
```

The default copy method for dataframes only copies one level deep, so basically it stores an array of columns.  If you assign elements of individual (column) arrays then, it can make changes to references to those arrays that exist elsewhere.

This method instead creates a new dataframe out of copies of the (column) arrays.

This is not named copy due to the fact that there is already an explicit copy(::DataTable) implementation in dataframes.

Note that deepcopy is recursive, so this is *NOT* the same thing as deepcopy(df), which  copies literally everything.

<a id='DatasToolbox.discreteDiff-Tuple{Array{T,1}}' href='#DatasToolbox.discreteDiff-Tuple{Array{T,1}}'>#</a>
**`DatasToolbox.discreteDiff`** &mdash; *Method*.



```
discreteDiff{T}(X::Array{T, 1})
```

Returns the discrete difference between adjacent elements of a time series.  So,  for instance, if one has a time series $y_{1},y_{2},ldots,y_{N}$ this will return a set of $δ$ such that $δ_{i} = y_{i+1} - y_{i}$.  The first element of the returned array will be a `NaN`.

<a id='DatasToolbox.featherRead-Tuple{AbstractString}' href='#DatasToolbox.featherRead-Tuple{AbstractString}'>#</a>
**`DatasToolbox.featherRead`** &mdash; *Method*.



```
featherRead(filename[; convert_strings=true])
```

A wrapper for reading dataframes which are saved in feather files.  The purpose of this wrapper is primarily for converting `WeakRefString` to `String`.  This will no longer be necessary in Julia 0.6.

<a id='DatasToolbox.featherWrite-Tuple{AbstractString,DataFrames.DataFrame}' href='#DatasToolbox.featherWrite-Tuple{AbstractString,DataFrames.DataFrame}'>#</a>
**`DatasToolbox.featherWrite`** &mdash; *Method*.



```
featherWrite(filename, df[, overwrite=false])
```

A wrapper for writing dataframes to feather files.  To be used while Feather.jl package is in development.

If `overwrite`, this will delete the existing file first (an extra step taken to avoid some strange bugs).

<a id='DatasToolbox.findBoundaryDict-Tuple{Array{T,1}}' href='#DatasToolbox.findBoundaryDict-Tuple{Array{T,1}}'>#</a>
**`DatasToolbox.findBoundaryDict`** &mdash; *Method*.



```
findBoundaryDict(X[, ncol; check_sort=true, comparator=DEFAULT])
```

For a sorted array `X`, find the boundaries between distinct values of column number `ncol`.  

<a id='DatasToolbox.fixColumnTypes!-Tuple{DataTables.DataTable}' href='#DatasToolbox.fixColumnTypes!-Tuple{DataTables.DataTable}'>#</a>
**`DatasToolbox.fixColumnTypes!`** &mdash; *Method*.



```
fixColumnTypes!(df)
```

Check to see if the dataframe `df` has any columns of type `Any` and attempt to convert them to the proper types.  This can be called from `convertPyDF` with the option `fixtypes`.

<a id='DatasToolbox.fixPyNones!-Tuple{DataTables.DataTable}' href='#DatasToolbox.fixPyNones!-Tuple{DataTables.DataTable}'>#</a>
**`DatasToolbox.fixPyNones!`** &mdash; *Method*.



```
fixPyNones!(df::DataTable)
```

Attempts to automatically convert all columns of a dataframe to have eltype `Any` while replacing all Python `None`s with `Nullable()`.

<a id='DatasToolbox.fixPyNones!-Tuple{Type{T},DataTables.DataTable,Symbol}' href='#DatasToolbox.fixPyNones!-Tuple{Type{T},DataTables.DataTable,Symbol}'>#</a>
**`DatasToolbox.fixPyNones!`** &mdash; *Method*.



```
fixPyNones!(dtype::DataType, df::DataTable, col::Symbol)
```

Attempts to convert a column of the dataframe to have eltype `dtype` while replacing all Python `None`s with `Nullable()`.

<a id='DatasToolbox.fixPyNones-Tuple{Type{T},NullableArrays.NullableArray}' href='#DatasToolbox.fixPyNones-Tuple{Type{T},NullableArrays.NullableArray}'>#</a>
**`DatasToolbox.fixPyNones`** &mdash; *Method*.



```
fixPyNones(dtype, a)
```

Attempts to convert a `NullableArray` to have eltype `dtype` while replacing all Python `None`s with `Nullable`.

<a id='DatasToolbox.generateSequence-Tuple{Function,DatasToolbox.TimeSeriesHandler{T},Integer}' href='#DatasToolbox.generateSequence-Tuple{Function,DatasToolbox.TimeSeriesHandler{T},Integer}'>#</a>
**`DatasToolbox.generateSequence`** &mdash; *Method*.



```
generateSequence(predict, dh, seq_length[, newcol_func; on_matrix=false])
```

Uses the supplied prediction function `predict` to generate a sequence of length `seq_length`. The sequence uses the end of the training dataset as initial input.

Note that this only makes sense when the output columns are a subset of the input columns.

If a function returning a dictionary or a dictionary of functions `newcol_func` is supplied, every time a new row of the input is generated, it will have columns specified by  `newcol_func`.  The dictionary should have keys equal to the column numbers of columns in the input matrix and values equal to functions that take a `Vector` (the previous input row) and output a new value for the column number given by the key.  The column numbers correspond to the index of the column in the specified input columns.

If `on_matrix` is true, the prediction function will take a matrix as input rather than a rank-3 tensor.

<a id='DatasToolbox.generateTest-Tuple{Function,DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.generateTest-Tuple{Function,DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.generateTest`** &mdash; *Method*.



```
generateTest(predict::Function, dh::TimeSeriesHandler; on_matrix::Bool=true)
```

Uses the supplied prediction function to attempt to predict the entire test set. Note that this assumes that the test set is ordered, sequential and immediately follows the training set.

See the documentation for `generateSequence`.

<a id='DatasToolbox.getCategoryVector' href='#DatasToolbox.getCategoryVector'>#</a>
**`DatasToolbox.getCategoryVector`** &mdash; *Function*.



```
getCategoryVector(A, vals[, dtype])
```

Get a vector which is 1 for each `a ∈ A` that satisfies `a ∈ vals`, and 0 otherwise. If `A` is a `NullableVector`, any null elements will be mapped to 0.

Optionally, one can specify the datatype of the output vector.

<a id='DatasToolbox.getDefaultCategoricalMapping-Tuple{Array}' href='#DatasToolbox.getDefaultCategoricalMapping-Tuple{Array}'>#</a>
**`DatasToolbox.getDefaultCategoricalMapping`** &mdash; *Method*.



```
getDefaultCategoricalMapping(A::Array)
```

Gets the default mapping of categorical variables which would be returned by numericalCategories.

<a id='DatasToolbox.getGroupedTestAnalysisData-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}' href='#DatasToolbox.getGroupedTestAnalysisData-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}'>#</a>
**`DatasToolbox.getGroupedTestAnalysisData`** &mdash; *Method*.



```
getGroupedTestAnalysisData(data, keycols[; names=[], squared_error=true])
getGroupedTestAnalysisData(dh, data, keycols[; names=[], squared_error=true])
getGroupedTestAnalysisData(gdh, data[; names=[], squared_error=true])
getGroupedTestAnalysisData(gdh, ŷ[; names=[], squared_error=true])
```

Groups the output of `getTestAnalysisData` by the columns `keycols`.  This is particularly useful for `GroupedDataHandler` where a typical use case is applying different estimators to different subsets of the data.  One can supply the output `getTestAnalysisData` as `data` or pass a `GroupedDataHandler` together with an output dictionary `ŷ`, in which case all the tables will be generated for you.

<a id='DatasToolbox.getMatrixDict-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}' href='#DatasToolbox.getMatrixDict-Tuple{DataTables.DataTable,Array{Symbol,1},Array{Symbol,1}}'>#</a>
**`DatasToolbox.getMatrixDict`** &mdash; *Method*.



```
getMatrixDict([T,] df, keycols, datacols)
```

Gets a dictionary the keys of which are the keys of a groupby of `df` by the columns `keycols` and the values of which are the matrices produced by taking `sdf[datacols]` of each `SubDataTable` `sdf` in the groupby.  Note that the keys are always tuples even if `keycols` only has one element.

If a type `T` is provided, the output matrices will be of type `Matrix{T}`.

<a id='DatasToolbox.getNormedHistogramData-Tuple{Array{T<:Real,1}}' href='#DatasToolbox.getNormedHistogramData-Tuple{Array{T<:Real,1}}'>#</a>
**`DatasToolbox.getNormedHistogramData`** &mdash; *Method*.



```
getNormedHistogramData(X)
```

Very annoyingly Gadfly does not yet support normed histograms.

This function returns an ordered pair of vectors which can be  fed to gadfly to create a normed histogram.  If the output is `m, w` one can do `plot(x=m, y=w, Geom.bar)` to create a histogram.

<a id='DatasToolbox.getRawTestTarget-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.getRawTestTarget-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.getRawTestTarget`** &mdash; *Method*.



```
getRawTestTarget(dh::TimeSeriesHandler)
```

Returns `y_test` directly from the dataframe for comparison with the output of generateTest.

<a id='DatasToolbox.getSquashedTestMatrix-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.getSquashedTestMatrix-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.getSquashedTestMatrix`** &mdash; *Method*.



```
getSquashedTestMatrix(dh::TimeSeriesHandler)
```

Gets a test input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_test is defined.

<a id='DatasToolbox.getSquashedTrainData-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.getSquashedTrainData-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.getSquashedTrainData`** &mdash; *Method*.



```
getSquashedTrainData(dh::TimeSeriesHandler; flatten::Bool=false)
```

Gets the training X, y pair where X is squashed using `getSquahdedTrainMatrix`. If `flatten`, also flatten `y`.

<a id='DatasToolbox.getSquashedTrainMatrix-Tuple{DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.getSquashedTrainMatrix-Tuple{DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.getSquashedTrainMatrix`** &mdash; *Method*.



```
getSquashedTrainMatrix(dh::TimeSeriesHandler)
```

Gets a training input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_train is defined.

<a id='DatasToolbox.getTestAnalysisData-Tuple{DatasToolbox.AbstractDH,Array}' href='#DatasToolbox.getTestAnalysisData-Tuple{DatasToolbox.AbstractDH,Array}'>#</a>
**`DatasToolbox.getTestAnalysisData`** &mdash; *Method*.



**`DataHandler`**

```
getTestAnalysisData(dh::AbstractDH, ŷ::Array; names::Vector{Symbol}=Symbol[],
                    squared_error::Bool=true)
```

Creates a dataframe from the test dataframe and a supplied prediction.  

The array names supplies the names for the columns, otherwise will generate default names.

Also generates error columns which are the difference between predictions and test data. If `squared_error`, will also create a column with squared error.

Note that this currently does nothing to handle transformations of the data.

<a id='DatasToolbox.getTestData-Tuple{DatasToolbox.AbstractDH}' href='#DatasToolbox.getTestData-Tuple{DatasToolbox.AbstractDH}'>#</a>
**`DatasToolbox.getTestData`** &mdash; *Method*.



```
getTestData(dh::AbstractDH; flatten::Bool=false)
```

Gets the test data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DatasToolbox.getTrainData-Tuple{DatasToolbox.AbstractDH}' href='#DatasToolbox.getTrainData-Tuple{DatasToolbox.AbstractDH}'>#</a>
**`DatasToolbox.getTrainData`** &mdash; *Method*.



```
getTrainData(dh::AbstractDH; flatten::Bool=false)
```

Gets the training data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DatasToolbox.getUnwrappedColumnElTypes' href='#DatasToolbox.getUnwrappedColumnElTypes'>#</a>
**`DatasToolbox.getUnwrappedColumnElTypes`** &mdash; *Function*.



```
getUnwrappedColumnElTypes(df[, cols=[]])
```

Get the element types of columns in a dataframe.  If the element types are `Nullable`,  instead give the `eltype` of the `Nullable`.  If `cols=[]` this will be done for all columns in the dataframe.

<a id='DatasToolbox.infast-Tuple{T,Array{T,1}}' href='#DatasToolbox.infast-Tuple{T,Array{T,1}}'>#</a>
**`DatasToolbox.infast`** &mdash; *Method*.



```
infast(x, collection)
```

Checks whether the object `x` is in `collection`. This is done efficiently by creating hashes for the objects in `collection`.  This should only be used if `collection` is large, as there is overhead in hashing and allocating.

<a id='DatasToolbox.nans2nulls-Tuple{NullableArrays.NullableArray{T,N}}' href='#DatasToolbox.nans2nulls-Tuple{NullableArrays.NullableArray{T,N}}'>#</a>
**`DatasToolbox.nans2nulls`** &mdash; *Method*.



```
nans2nulls(col)
nans2nulls(df, colname)
```

Converts all `NaN`s appearing in the column to `Nullable()`.  The return type is `NullableArray`, even if the original type of the column is not.

<a id='DatasToolbox.numericalCategories!-Tuple{Type{T},DataTables.DataTable,Array{Symbol,N}}' href='#DatasToolbox.numericalCategories!-Tuple{Type{T},DataTables.DataTable,Array{Symbol,N}}'>#</a>
**`DatasToolbox.numericalCategories!`** &mdash; *Method*.



```
numericalCategories!(otype::DataType, df::DataTable, cols::Array{Symbol})
```

Converts categorical variables into numerical values for multiple columns in a dataframe.  

**TODO** For now doesn't return mapping, may have to implement some type of  mapping type.

<a id='DatasToolbox.numericalCategories!-Tuple{Type{T},DataTables.DataTable,Symbol}' href='#DatasToolbox.numericalCategories!-Tuple{Type{T},DataTables.DataTable,Symbol}'>#</a>
**`DatasToolbox.numericalCategories!`** &mdash; *Method*.



```
numericalCategories!(otype::DataType, df::DataTable, col::Symbol)
```

Converts a categorical value in a column into a numerical variable of the given type.

Returns the mapping.

<a id='DatasToolbox.numericalCategories-Tuple{Type{T},Array}' href='#DatasToolbox.numericalCategories-Tuple{Type{T},Array}'>#</a>
**`DatasToolbox.numericalCategories`** &mdash; *Method*.



```
numericalCategories(otype, A)
```

Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array so it always maps to an integer

<a id='DatasToolbox.outer-Tuple{Array{T,1},Array{T,1}}' href='#DatasToolbox.outer-Tuple{Array{T,1},Array{T,1}}'>#</a>
**`DatasToolbox.outer`** &mdash; *Method*.



```
outer(A, B)
```

Performs the outer product of two tensors A_{i₁…iₙ}B_{j₁…jₙ}.

**TODO** Currently only implemented for A and B as vectors.

<a id='DatasToolbox.pandas-Tuple{DataTables.DataTable}' href='#DatasToolbox.pandas-Tuple{DataTables.DataTable}'>#</a>
**`DatasToolbox.pandas`** &mdash; *Method*.



```
pandas(df)
```

Convert a dataframe to a pandas pyobject.

<a id='DatasToolbox.pickle-Tuple{String,Any}' href='#DatasToolbox.pickle-Tuple{String,Any}'>#</a>
**`DatasToolbox.pickle`** &mdash; *Method*.



```
pickle(filename, object)
```

Converts the provided object to a PyObject and serializes it in the python pickle format.  If the object provided is a `DataTable`, this will first convert it to a pandas dataframe.

<a id='DatasToolbox.pwd2PyPath-Tuple{}' href='#DatasToolbox.pwd2PyPath-Tuple{}'>#</a>
**`DatasToolbox.pwd2PyPath`** &mdash; *Method*.



```
pwd2PyPath()
```

Adds the present working directory to the python path variable.

<a id='DatasToolbox.randomData-Tuple{Vararg{DataType,N}}' href='#DatasToolbox.randomData-Tuple{Vararg{DataType,N}}'>#</a>
**`DatasToolbox.randomData`** &mdash; *Method*.



```
randomData(dtypes...; nrows=10^4)
```

Creates a random dataframe with columns of types specified by `dtypes`.  This is useful for testing various dataframe related functionality.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,AbstractFloat}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,AbstractFloat}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
       assign::Bool=true)
```

Creates a train, test split by fraction.  The fraction given is the test fraction.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,BitArray}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,BitArray}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, constraint::BitArray)
```

Splits the data into training and test sets using a BitArray that must correspond to elements of dh.df.  The elements of the dataframe for which the BitArray holds 1 will be in the test  set, the remaining elements will be in the training set.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,Integer}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH,Integer}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, index::Integer; assign::Bool=true)
```

Creates a train, test split by index.  The index given is the last index of the training set. If `assign`, this will assign the training and test data.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.TimeSeriesHandler,Integer}' href='#DatasToolbox.split!-Tuple{DatasToolbox.TimeSeriesHandler,Integer}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)
```

Splits the data by time-index.  All datapoints with τ up to and including the timeindex given (τ₀) will be in the training set, while all those with τ > τ₀ will be in  the test set.

<a id='DatasToolbox.splitByNSequences!-Tuple{DatasToolbox.TimeSeriesHandler,Integer}' href='#DatasToolbox.splitByNSequences!-Tuple{DatasToolbox.TimeSeriesHandler,Integer}'>#</a>
**`DatasToolbox.splitByNSequences!`** &mdash; *Method*.



```
splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                   assign::Bool=true, sort::Bool=true)
```

Splits the dataframe by the number of sequences in the test set.  This does nothing to account for the possibility of missing data.

Note that the actual number of usable test sequences in the resulting test set is of course greater than n_sequences.

<a id='DatasToolbox.subMatricesByClass-Tuple{Array{T,2},U,Integer}' href='#DatasToolbox.subMatricesByClass-Tuple{Array{T,2},U,Integer}'>#</a>
**`DatasToolbox.subMatricesByClass`** &mdash; *Method*.



```
subMatricesByClass(X[, y], ncol)
```

Breaks the arrays `X` and `y` (optional) into dictionaries of `class=>submatrix`  pairs where `class` is one of the distinct values of the column `ncol` of `X` and  `submatrix` is the range of rows of `X` or `y` where `X[:,ncol]` has the value `class`.

This function is intended for breaking up training and test sets to be used with sets of different models.

<a id='DatasToolbox.trainOnMatrix-Tuple{Function,DatasToolbox.TimeSeriesHandler}' href='#DatasToolbox.trainOnMatrix-Tuple{Function,DatasToolbox.TimeSeriesHandler}'>#</a>
**`DatasToolbox.trainOnMatrix`** &mdash; *Method*.



```
trainOnMatrix(train::Function, dh::TimeSeriesHandler; flatten::Bool=true)
```

Trains an object designed to take a matrix (as opposed to a rank-3 tensor) as input. The first argument is the function used to train.

If flatten, the training output is converted to a vector.  Most methods that take matrix input take vector target data.

Assumes the function takes input of the form train(X, y).

<a id='DatasToolbox.unnormalize!-Tuple{DatasToolbox.AbstractDH{T},Array{T,2},Array{Symbol,1}}' href='#DatasToolbox.unnormalize!-Tuple{DatasToolbox.AbstractDH{T},Array{T,2},Array{Symbol,1}}'>#</a>
**`DatasToolbox.unnormalize!`** &mdash; *Method*.



```
unnormalize!{T}(dh::AbstractDH{T}, X::Matrix{T}, cols::Vector{Symbol})
```

Performs the inverse of the centering and rescaling operations on a matrix. This can also be called on a single column with a `Symbol` as the last argument.

<a id='DatasToolbox.unpickle-Tuple{String}' href='#DatasToolbox.unpickle-Tuple{String}'>#</a>
**`DatasToolbox.unpickle`** &mdash; *Method*.



```
unpickle([dtype,] filename[, fixtypes=true])
```

Deserializes a python pickle file and returns the object it contains. Additionally, if `DataTable` is given as the first argument, will attempt to convert the object to a Julia dataframe with the flag `fixtypes` (see `convertPyDF`).

<a id='DatasToolbox.@constrain-Tuple{Any,Any}' href='#DatasToolbox.@constrain-Tuple{Any,Any}'>#</a>
**`DatasToolbox.@constrain`** &mdash; *Macro*.



```
@constrain(df, expr)
```

Constrains the dataframe to rows for which `expr` evaluates to `true`.  `expr` should specify columns with column names written as symbols.  For example, to do `(a ∈ A) > M` one should write `:A .> M`.

<a id='DatasToolbox.@info-Tuple{String,Any}' href='#DatasToolbox.@info-Tuple{String,Any}'>#</a>
**`DatasToolbox.@info`** &mdash; *Macro*.



```
@info code
```

Executes code sandwhiched between informative info messages telling the user that the code is being executed.

<a id='DatasToolbox.@infotime-Tuple{String,Any}' href='#DatasToolbox.@infotime-Tuple{String,Any}'>#</a>
**`DatasToolbox.@infotime`** &mdash; *Macro*.



```
@infotime code
```

Executes code sandwhiched between informative info messages telling the user that the code is being executed, while applying the `@time` macro to the code.

<a id='DatasToolbox.@pyslice-Tuple{Any}' href='#DatasToolbox.@pyslice-Tuple{Any}'>#</a>
**`DatasToolbox.@pyslice`** &mdash; *Macro*.



```
@pyslice slice
```

Gets a slice of a python object.  Does not shift indices.

<a id='DatasToolbox.@selectTest!-Tuple{Any,Any}' href='#DatasToolbox.@selectTest!-Tuple{Any,Any}'>#</a>
**`DatasToolbox.@selectTest!`** &mdash; *Macro*.



```
@selectTrain!(dh, expr)
```

Set the test set to be the subset of the `DataHandler`'s dataframe for which `expr` is true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding to column names for values.  See the documentation for `@constrain` and `@split!`  for examples.

<a id='DatasToolbox.@selectTrain!-Tuple{Any,Any}' href='#DatasToolbox.@selectTrain!-Tuple{Any,Any}'>#</a>
**`DatasToolbox.@selectTrain!`** &mdash; *Macro*.



```
@selectTrain!(dh, expr)
```

Set the training set to be the subset of the `DataHandler`'s dataframe for which `expr` is true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding to column names for values.  See the documentation for `@constrain` and `@split!`  for examples.

<a id='DatasToolbox.@split!-Tuple{Any,Any}' href='#DatasToolbox.@split!-Tuple{Any,Any}'>#</a>
**`DatasToolbox.@split!`** &mdash; *Macro*.



```
@split!(dh, expr)
```

Splits the `DataHandler`'s DataTable into training in test set, such that the test set is the set of datapoints for which `expr` is true, and the training set is the set of datapoints for which `expr` is false.  `expr` should be an expression that evaluates to `Bool` with symbols in place of column names.  For example, if the columns of the dataframe are `[:x1, :x2, :x3]` one can do `expr = (:x1 > 0.0) | (tanh(:x3) > e^6)`.

See documentation on the `@constrain` macro which `@split!` calls internally.

