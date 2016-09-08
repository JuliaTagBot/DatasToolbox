
<a id='DatasToolbox-1'></a>

# DatasToolbox


This is a set of tools for performing common, generic manipulations on data, primarily for  machine learning and optimization.  It also contains a set of tools for easily converting from Python in a type-correct way.


Note that there is currently a bug in the documentation generator that causes it to also list private methods.  All such methods have names that start with `_`.


<a id='API-Docs-1'></a>

## API Docs

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

<a id='Base.Random.shuffle!-Tuple{DataFrames.DataFrame}' href='#Base.Random.shuffle!-Tuple{DataFrames.DataFrame}'>#</a>
**`Base.Random.shuffle!`** &mdash; *Method*.



```
shuffle!(df::DataFrame)
```

Shuffles a dataframe in place.

<a id='Base.Random.shuffle!-Tuple{DatasToolbox.AbstractDH{T}}' href='#Base.Random.shuffle!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
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

<a id='Base.convert-Tuple{Union{Type{Int32},Type{Int64}},DataArrays.DataArray{Float32,1}}' href='#Base.convert-Tuple{Union{Type{Int32},Type{Int64}},DataArrays.DataArray{Float32,1}}'>#</a>
**`Base.convert`** &mdash; *Method*.



```
convert(dtype::Union{Type{Int32}, Type{Int64}}, a::DataArray)
```

This converts a column of floats that should have been ints, but got converted to floats because it has missing values which were converted to NaN's. The supplied `DataArray` should have eltype `Float32` or `Float64`.

<a id='DatasToolbox._fixBadPyConversions-Tuple{PyCall.PyObject,AbstractString}' href='#DatasToolbox._fixBadPyConversions-Tuple{PyCall.PyObject,AbstractString}'>#</a>
**`DatasToolbox._fixBadPyConversions`** &mdash; *Method*.



Checks to see if the column is one of the types known to fuck up conversions. If so, makes the appropriate changes.

<a id='DatasToolbox._get_assign_data-Tuple{Symbol,DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox._get_assign_data-Tuple{Symbol,DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox._get_assign_data`** &mdash; *Method*.



Used by assignTrain!, and assignTest!

<a id='DatasToolbox._get_assign_data_parallel-Tuple{Symbol,DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox._get_assign_data_parallel-Tuple{Symbol,DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox._get_assign_data_parallel`** &mdash; *Method*.



Parallel version, used by assignTrain! and assignTest!.

<a id='DatasToolbox._replaceExprColNames!-Tuple{Expr,DatasToolbox.AbstractDH{T},Symbol}' href='#DatasToolbox._replaceExprColNames!-Tuple{Expr,DatasToolbox.AbstractDH{T},Symbol}'>#</a>
**`DatasToolbox._replaceExprColNames!`** &mdash; *Method*.



Used by constrain and split macros.  Looks through expressions for symbols of  columns, and replaces them with proper ref calls.

<a id='DatasToolbox._squashXTensor-Tuple{Array{T,N},DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox._squashXTensor-Tuple{Array{T,N},DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox._squashXTensor`** &mdash; *Method*.



Squashes the last two indices of a rank-3 tensor into a matrix.  For internal use.

<a id='DatasToolbox.assign!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.assign!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.assign!`** &mdash; *Method*.



```
assign!(dh::AbstractDH)
```

Assigns training and test data in the data handler.

<a id='DatasToolbox.assign!-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.assign!-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
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

Note that this is silent if the test dataframe is empty.

<a id='DatasToolbox.assignTest!-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.assignTest!-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.assignTest!`** &mdash; *Method*.



```
assignTest!(dh::TimeSeriesHandler; sort::Bool=true)
```

Assigns the test data.  X output will be of shape (samples, seq_length, seq_width). One should be extremely careful if not sorting.

Note that in the time series case this isn't very useful.  One should instead use one of the assigned prediction functions.

Note that this is silent if the dataframe is empty.

<a id='DatasToolbox.assignTrain!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.assignTrain!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh::AbstractDH)
```

Assigns the training data in the data handler so it can be retrieved in proper form.

Note that this is silent if the training dataframe is empty.

<a id='DatasToolbox.assignTrain!-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.assignTrain!-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.assignTrain!`** &mdash; *Method*.



```
assignTrain!(dh::TimeSeriesHandler; sort::Bool=true, parallel::Bool=false)
```

Assigns the training data.  X output will be of shape (samples, seq_length, seq_width). If `sort` is true, will sort the dataframe first.  One should be extremely careful if `sort` is false.  If `parallel` is true the data will be generated in parallel (using  workers, not threads).  This is useful because this data manipulation is complicated and potentially slow.

Note that this is silent if the dataframe is empty.

**TODO** I'm pretty sure the parallel version isn't working right because it doesn't use shared arrays.  Revisit in v0.5 with threads.

<a id='DatasToolbox.canNormalize-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.canNormalize-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.canNormalize`** &mdash; *Method*.



```
canNormalize(dh::AbstractDH)
```

Determines whether the data in the datahandler can be normlized, i.e. because the parameters have or haven't been computed yet.

<a id='DatasToolbox.computeNormalizeParameters!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.computeNormalizeParameters!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.computeNormalizeParameters!`** &mdash; *Method*.



```
computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
```

Gets the parameters for centering and rescaling from either the training dataset  (`dataset=:dfTrain`) or the test dataset (`dataset=:dfTest`).

Does this using the training dataframe by default, but can be set to use test. Exits normally if this doesn't need to be done for any columns.

This should always be called before `normalize!`, that way you have control over what dataset the parameters are computed from.

<a id='DatasToolbox.convertCol-Tuple{DataFrames.DataFrame,Symbol,DataType}' href='#DatasToolbox.convertCol-Tuple{DataFrames.DataFrame,Symbol,DataType}'>#</a>
**`DatasToolbox.convertCol`** &mdash; *Method*.



```
convertCol(df::DataFrame, col::Symbol, dtype::DataType)
```

Converts a column, possibly containing python objects, to a column with eltype `dtype`. The column itself will be a `DataArray` with `NA` values inserted where Python `None`s are found.  Note that this isn't terribly efficient because it has to check for `None`s.

<a id='DatasToolbox.convertNulls!-Tuple{Array{T,1},T}' href='#DatasToolbox.convertNulls!-Tuple{Array{T,1},T}'>#</a>
**`DatasToolbox.convertNulls!`** &mdash; *Method*.



```
convertNulls!{T}(A::Array{T, 1}, newvalue::T)
```

Converts all null values (NaN's and Nullable()) to a particular value. Note this has to check whether the type is Nullable.

<a id='DatasToolbox.convertNulls!-Tuple{DataFrames.DataFrame,Array{Symbol,1},Any}' href='#DatasToolbox.convertNulls!-Tuple{DataFrames.DataFrame,Array{Symbol,1},Any}'>#</a>
**`DatasToolbox.convertNulls!`** &mdash; *Method*.



```
convertNulls!(df::DataFrame, cols::Vector{Symbol}, newvalue::Any)
```

Convert all null values in columns of a DataFrame to a particular value.

There is also a method for passing a single column symbol, not as a vector.

<a id='DatasToolbox.convertNulls-Tuple{DataArrays.DataArray{T,N},T}' href='#DatasToolbox.convertNulls-Tuple{DataArrays.DataArray{T,N},T}'>#</a>
**`DatasToolbox.convertNulls`** &mdash; *Method*.



```
convertNulls{T}(A::DataArray{T}, newvalue::T)
```

Converts all null vlaues (NA's, NaN's and Nullable()) to a particular value.

<a id='DatasToolbox.convertPyDF-Tuple{PyCall.PyObject}' href='#DatasToolbox.convertPyDF-Tuple{PyCall.PyObject}'>#</a>
**`DatasToolbox.convertPyDF`** &mdash; *Method*.



```
convertPyDF(df::PyObject; migrate::Bool=true, fix_nones::Bool=true)
```

Converts a pandas dataframe to a Julia dataframe.  If `migrate` is true this will try to properly assign types to columns.  If `fix_nones` is true, this will check for columns which have eltype `PyObject` and convert them to have eltype `Any`, replacing all Python `None`s with `NA`.

<a id='DatasToolbox.copyColumns-Tuple{DataFrames.DataFrame}' href='#DatasToolbox.copyColumns-Tuple{DataFrames.DataFrame}'>#</a>
**`DatasToolbox.copyColumns`** &mdash; *Method*.



```
copyColumns(df::DataFrame)
```

The default copy method for dataframes only copies one level deep, so basically it stores an array of columns.  If you assign elements of individual (column) arrays then, it can make changes to references to those arrays that exist elsewhere.

This method instead creates a new dataframe out of copies of the (column) arrays.

This is not named copy due to the fact that there is already an explicit copy(::DataFrame) implementation in dataframes.

Note that deepcopy is recursive, so this is *NOT* the same thing as deepcopy(df), which  copies literally everything.

<a id='DatasToolbox.discreteDiff-Tuple{Array{T,1}}' href='#DatasToolbox.discreteDiff-Tuple{Array{T,1}}'>#</a>
**`DatasToolbox.discreteDiff`** &mdash; *Method*.



```
discreteDiff{T}(X::Array{T, 1})
```

Returns the discrete difference between adjacent elements of a time series.  So,  for instance, if one has a time series `y_{1},y_{2},ldots,y_{N}` this will return a set of `δ` such that `δ_{i} = y_{i+1} - y_{i}`.  The first element of the returned array will be a `NaN`.

<a id='DatasToolbox.fixPyNones!-Tuple{DataFrames.DataFrame}' href='#DatasToolbox.fixPyNones!-Tuple{DataFrames.DataFrame}'>#</a>
**`DatasToolbox.fixPyNones!`** &mdash; *Method*.



```
fixPyNones!(df::DataFrame)
```

Attempts to automatically convert all columns of a dataframe to have eltype `Any` while replacing all Python `None`s with `NA`.

<a id='DatasToolbox.fixPyNones!-Tuple{DataType,DataFrames.DataFrame,Symbol}' href='#DatasToolbox.fixPyNones!-Tuple{DataType,DataFrames.DataFrame,Symbol}'>#</a>
**`DatasToolbox.fixPyNones!`** &mdash; *Method*.



```
fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)
```

Attempts to convert a column of the dataframe to have eltype `dtype` while replacing all Python `None`s with `NA`.

<a id='DatasToolbox.fixPyNones-Tuple{DataType,DataArrays.DataArray{T,N}}' href='#DatasToolbox.fixPyNones-Tuple{DataType,DataArrays.DataArray{T,N}}'>#</a>
**`DatasToolbox.fixPyNones`** &mdash; *Method*.



```
fixPyNones(dtype::DataType, a::DataArray)
```

Attempts to convert a `DataArray` to have eltype `dtype` while replacing all Python `None`s with `NA`.

<a id='DatasToolbox.generateSequence' href='#DatasToolbox.generateSequence'>#</a>
**`DatasToolbox.generateSequence`** &mdash; *Function*.



```
generateSequence(predict::Function, dh::TimeSeriesHandler,
                 seq_length::Integer=8; on_matrix::Bool=false)
```

Uses the supplied prediction function to generate a sequence of a specified length. The sequence uses the end of the training dataset as initial input.

Note that this only makes sense if the intersection between the input and output columns isn't ∅.  For now we enforce that they must be identical.

If `on_matrix` is true, the prediction function will take a matrix as input rather than a rank-3 tensor.

**TODO** Fix this so that it works for arbitrary input, output.

<a id='DatasToolbox.generateTest-Tuple{Function,DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.generateTest-Tuple{Function,DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.generateTest`** &mdash; *Method*.



```
generateTest(predict::Function, dh::TimeSeriesHandler; on_matrix::Bool=true)
```

Uses the supplied prediction function to attempt to predict the entire test set. Note that this assumes that the test set is ordered, sequential and immediately follows the training set.

See the documentation for `generateSequence`.

<a id='DatasToolbox.getDefaultCategoricalMapping-Tuple{Array{T,N}}' href='#DatasToolbox.getDefaultCategoricalMapping-Tuple{Array{T,N}}'>#</a>
**`DatasToolbox.getDefaultCategoricalMapping`** &mdash; *Method*.



```
getDefaultCategoricalMapping(A::Array)
```

Gets the default mapping of categorical variables which would be returned by numericalCategories.

<a id='DatasToolbox.getRawTestTarget-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.getRawTestTarget-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.getRawTestTarget`** &mdash; *Method*.



```
getRawTestTarget(dh::TimeSeriesHandler)
```

Returns `y_test` directly from the dataframe for comparison with the output of generateTest.

<a id='DatasToolbox.getSquashedTestMatrix-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.getSquashedTestMatrix-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.getSquashedTestMatrix`** &mdash; *Method*.



```
getSquashedTestMatrix(dh::TimeSeriesHandler)
```

Gets a test input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_test is defined.

<a id='DatasToolbox.getSquashedTrainData-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.getSquashedTrainData-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.getSquashedTrainData`** &mdash; *Method*.



```
getSquashedTrainData(dh::TimeSeriesHandler; flatten::Bool=false)
```

Gets the training X, y pair where X is squashed using `getSquahdedTrainMatrix`. If `flatten`, also flatten `y`.

<a id='DatasToolbox.getSquashedTrainMatrix-Tuple{DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.getSquashedTrainMatrix-Tuple{DatasToolbox.TimeSeriesHandler{T}}'>#</a>
**`DatasToolbox.getSquashedTrainMatrix`** &mdash; *Method*.



```
getSquashedTrainMatrix(dh::TimeSeriesHandler)
```

Gets a training input tensor in which all the inputs are arranged along a single axis (i.e. in a matrix).

Assumes the handler's X_train is defined.

<a id='DatasToolbox.getTestAnalysisData-Tuple{DatasToolbox.AbstractDH{T},Array{T,N}}' href='#DatasToolbox.getTestAnalysisData-Tuple{DatasToolbox.AbstractDH{T},Array{T,N}}'>#</a>
**`DatasToolbox.getTestAnalysisData`** &mdash; *Method*.



```
getTestAnalysisData(dh::AbstractDH, ŷ::Array; names::Vector{Symbol}=Symbol[],
                    squared_error::Bool=true)
```

Creates a dataframe from the test dataframe and a supplied prediction.  

The array names supplies the names for the columns, otherwise will generate default names.

Also generates error columns which are the difference between predictions and test data. If `squared_error`, will also create a column with squared error.

Note that this currently does nothing to handle transformations of the data.

<a id='DatasToolbox.getTestData-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.getTestData-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.getTestData`** &mdash; *Method*.



```
getTestData(dh::AbstractDH; flatten::Bool=false)
```

Gets the test data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DatasToolbox.getTrainData-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.getTrainData-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.getTrainData`** &mdash; *Method*.



```
getTrainData(dh::AbstractDH; flatten::Bool=false)
```

Gets the training data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.

<a id='DatasToolbox.loadPickledDF-Tuple{AbstractString}' href='#DatasToolbox.loadPickledDF-Tuple{AbstractString}'>#</a>
**`DatasToolbox.loadPickledDF`** &mdash; *Method*.



```
loadPickledDF(filename::AbstractString; migrate::Bool=true, fix_nones::Bool=true)
```

Loads a pickled python dataframe, converting it to a Julia dataframe using `convertPyDF`.

<a id='DatasToolbox.migrateTypes!-Tuple{DataFrames.DataFrame}' href='#DatasToolbox.migrateTypes!-Tuple{DataFrames.DataFrame}'>#</a>
**`DatasToolbox.migrateTypes!`** &mdash; *Method*.



```
migrateTypes!(df::DataFrame)
```

Attempts to convert all columns of a dataframe to the proper type based on the Python types found in it.   **TODO** Right now this just checks the first elements in the column.  Also, will have to change `ASCIIString` conversion to `String` for Julia v0.5.

<a id='DatasToolbox.normalize!-Tuple{DatasToolbox.AbstractDH{T}}' href='#DatasToolbox.normalize!-Tuple{DatasToolbox.AbstractDH{T}}'>#</a>
**`DatasToolbox.normalize!`** &mdash; *Method*.



```
normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
normalizeTrain!(dh::AbstractDH)
normalizeTest!(dh::AbstractDH)
```

Centers and rescales the columns set by `normalize_cols` in the `DataHandler` constructor.

<a id='DatasToolbox.numericalCategories!-Tuple{DataType,DataFrames.DataFrame,Array{Symbol,N}}' href='#DatasToolbox.numericalCategories!-Tuple{DataType,DataFrames.DataFrame,Array{Symbol,N}}'>#</a>
**`DatasToolbox.numericalCategories!`** &mdash; *Method*.



```
numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol})
```

Converts categorical variables into numerical values for multiple columns in a dataframe.  

**TODO** For now doesn't return mapping, may have to implement some type of  mapping type.

<a id='DatasToolbox.numericalCategories!-Tuple{DataType,DataFrames.DataFrame,Symbol}' href='#DatasToolbox.numericalCategories!-Tuple{DataType,DataFrames.DataFrame,Symbol}'>#</a>
**`DatasToolbox.numericalCategories!`** &mdash; *Method*.



```
numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)
```

Converts a categorical value in a column into a numerical variable of the given type.

Returns the mapping.

<a id='DatasToolbox.numericalCategories-Tuple{DataType,Array{T,N}}' href='#DatasToolbox.numericalCategories-Tuple{DataType,Array{T,N}}'>#</a>
**`DatasToolbox.numericalCategories`** &mdash; *Method*.



```
numericalCategories(otype::DataType, A::Array)
```

Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array so it always maps to an integer

<a id='DatasToolbox.pwd2PyPath-Tuple{}' href='#DatasToolbox.pwd2PyPath-Tuple{}'>#</a>
**`DatasToolbox.pwd2PyPath`** &mdash; *Method*.



```
pwd2PyPath()
```

Adds the present working directory to the python path variable.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},AbstractFloat}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},AbstractFloat}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
       assign::Bool=true)
```

Creates a train, test split by fraction.  The fraction given is the test fraction.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},BitArray{N}}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},BitArray{N}}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, constraint::BitArray)
```

Splits the data into training and test sets using a BitArray that must correspond to elements of dh.df.  The elements of the dataframe for which the BitArray holds 1 will be in the test  set, the remaining elements will be in the training set.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},Integer}' href='#DatasToolbox.split!-Tuple{DatasToolbox.AbstractDH{T},Integer}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::AbstractDH, index::Integer; assign::Bool=true)
```

Creates a train, test split by index.  The index given is the last index of the training set. If `assign`, this will assign the training and test data.

<a id='DatasToolbox.split!-Tuple{DatasToolbox.TimeSeriesHandler{T},Integer}' href='#DatasToolbox.split!-Tuple{DatasToolbox.TimeSeriesHandler{T},Integer}'>#</a>
**`DatasToolbox.split!`** &mdash; *Method*.



```
split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)
```

Splits the data by time-index.  All datapoints with τ up to and including the timeindex given (τ₀) will be in the training set, while all those with τ > τ₀ will be in  the test set.

<a id='DatasToolbox.splitByNSequences!-Tuple{DatasToolbox.TimeSeriesHandler{T},Integer}' href='#DatasToolbox.splitByNSequences!-Tuple{DatasToolbox.TimeSeriesHandler{T},Integer}'>#</a>
**`DatasToolbox.splitByNSequences!`** &mdash; *Method*.



```
splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                   assign::Bool=true, sort::Bool=true)
```

Splits the dataframe by the number of sequences in the test set.  This does nothing to account for the possibility of missing data.

Note that the actual number of usable test sequences in the resulting test set is of course greater than n_sequences.

<a id='DatasToolbox.trainOnMatrix-Tuple{Function,DatasToolbox.TimeSeriesHandler{T}}' href='#DatasToolbox.trainOnMatrix-Tuple{Function,DatasToolbox.TimeSeriesHandler{T}}'>#</a>
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

<a id='DatasToolbox.@constrain!' href='#DatasToolbox.@constrain!'>#</a>
**`DatasToolbox.@constrain!`** &mdash; *Macro*.



```
@constrain! dh traintest constraint
```

Constrains either the test or training dataframe.  See the documentation for those.

Note that these constraints are applied directly to the train or test dataframe, so some of the columns may be transformed in some way.

For example: `@constrain dh test x .≥ 3`

**TODO** This still isn't working right.  Will probably have to wait for v0.5.

<a id='DatasToolbox.@constrainTest!' href='#DatasToolbox.@constrainTest!'>#</a>
**`DatasToolbox.@constrainTest!`** &mdash; *Macro*.



```
@constrainTest dh constraint
```

Constrains the test dataframe to satisfy the provided constriant.

Input should be in the form of ColumnName relation value.  For example, `x .> 3` imposes `df[:x] .> 3`.

<a id='DatasToolbox.@constrainTrain!' href='#DatasToolbox.@constrainTrain!'>#</a>
**`DatasToolbox.@constrainTrain!`** &mdash; *Macro*.



```
@constrainTrain dh constraint
```

Constrains the training dataframe to satisfy the provided constraint.

Input should be in the form of ColumnName relation value.  For example `x .> 3` imposes `df[:x] .> 3`.

<a id='DatasToolbox.@pyslice' href='#DatasToolbox.@pyslice'>#</a>
**`DatasToolbox.@pyslice`** &mdash; *Macro*.



```
@pyslice slice
```

Gets a slice of a python object.  Does not shift indices.

<a id='DatasToolbox.@split!' href='#DatasToolbox.@split!'>#</a>
**`DatasToolbox.@split!`** &mdash; *Macro*.



```
@split! dh constraint
```

Splits the data into training and test sets.  The test set will be the data for which the provided constraint is true.

For example, `@split dh x .≥ 3.0` will set the test set to be `df[:x] .≥ 3.0` and the training set to be `df[:x] .< 3.0`.

**NOTE** This still isn't working right, will probably have to wait for v0.5.

