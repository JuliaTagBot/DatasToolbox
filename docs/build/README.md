
<a id='DatasToolbox-1'></a>

# DatasToolbox


This is a set of tools for performing common, generic manipulations on data, primarily for  machine learning and optimization.  It also contains a set of tools for easily converting from Python in a type-correct way.


Note that there is currently a bug in the documentation generator that causes it to also list private methods.  All such methods have names that start with `_`.


<a id='API-Docs-1'></a>

## API Docs

<a id='Base.Random.shuffle!-Tuple{DataFrames.DataFrame}' href='#Base.Random.shuffle!-Tuple{DataFrames.DataFrame}'>#</a>
**`Base.Random.shuffle!`** &mdash; *Method*.



```
shuffle!(df::DataFrame)
```

Shuffles a dataframe in place.

<a id='Base.convert-Tuple{Union{Type{Int32},Type{Int64}},DataArrays.DataArray{Float32,1}}' href='#Base.convert-Tuple{Union{Type{Int32},Type{Int64}},DataArrays.DataArray{Float32,1}}'>#</a>
**`Base.convert`** &mdash; *Method*.



```
convert(dtype::Union{Type{Int32}, Type{Int64}}, a::DataArray)
```

This converts a column of floats that should have been ints, but got converted to floats because it has missing values which were converted to NaN's. The supplied `DataArray` should have eltype `Float32` or `Float64`.

<a id='DatasToolbox._fixBadPyConversions-Tuple{PyCall.PyObject,AbstractString}' href='#DatasToolbox._fixBadPyConversions-Tuple{PyCall.PyObject,AbstractString}'>#</a>
**`DatasToolbox._fixBadPyConversions`** &mdash; *Method*.



Checks to see if the column is one of the types known to fuck up conversions. If so, makes the appropriate changes.

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

<a id='DatasToolbox.getDefaultCategoricalMapping-Tuple{Array{T,N}}' href='#DatasToolbox.getDefaultCategoricalMapping-Tuple{Array{T,N}}'>#</a>
**`DatasToolbox.getDefaultCategoricalMapping`** &mdash; *Method*.



```
getDefaultCategoricalMapping(A::Array)
```

Gets the default mapping of categorical variables which would be returned by numericalCategories.

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

<a id='DatasToolbox.discreteDiff-Tuple{Array{T,1}}' href='#DatasToolbox.discreteDiff-Tuple{Array{T,1}}'>#</a>
**`DatasToolbox.discreteDiff`** &mdash; *Method*.



```
discreteDiff{T}(X::Array{T, 1})
```

Returns the discrete difference between adjacent elements of a time series.  So,  for instance, if one has a time series `y_{1},y_{2},ldots,y_{N}` this will return a set of `δ` such that `δ_{i} = y_{i+1} - y_{i}`.  The first element of the returned array will be a `NaN`.

<a id='DatasToolbox.pwd2PyPath-Tuple{}' href='#DatasToolbox.pwd2PyPath-Tuple{}'>#</a>
**`DatasToolbox.pwd2PyPath`** &mdash; *Method*.



```
pwd2PyPath()
```

Adds the present working directory to the python path variable.

<a id='DatasToolbox.@pyslice' href='#DatasToolbox.@pyslice'>#</a>
**`DatasToolbox.@pyslice`** &mdash; *Macro*.



```
@pyslice slice
```

Gets a slice of a python object.  Does not shift indices.

