
"""
Converts a dataframe column to a date.
"""
function convertCol(df::DataFrame, col::Symbol, dtype::DataType)
    return map(d -> convert(dtype, d), df[col])
end
export convertCol


# TODO for now just checks the first element of each column
"""
Attempts to convert all columns of a dataframe to Julia types.
"""
function migrateTypes!(df::DataFrame)
    @pyimport datetime
    @pyimport decimal
    for col in names(df)
        isinstance = pybuiltin("isinstance")
        pystr = pybuiltin("str")
        if isinstance(df[1, col], pystr)
            df[col] = convertCol(df, col, ASCIIString)
        elseif isinstance(df[1, col], datetime.date)
            df[col] = convertCol(df, col, Date)
        elseif isinstance(df[1, col], datetime.datetime)
            df[col] = convertCol(df, col, DateTime)
        # consider removing this, it's obscure and probably shouldn't always be checked
        elseif isinstance(df[1, col], decimal.Decimal)
            df[col] = convertCol(df, col, Float32)
        end
    end
end
export migrateTypes!
        

"""
A function for converting python dataframes to Julia dataframes.
Note that dates are not automatically converted.  You can specify which
columns are dates with the datecols argument.
"""
function convertPyDF(df::PyObject; 
                     migrate::Bool=true)
    jdf = DataFrame()
    for col in df[:columns]
        jdf[symbol(col)] = df[col][:values]
    end
    if migrate migrateTypes!(jdf) end
    return jdf
end
export convertPyDF


"""
Loads a pickled python dataframe.
"""
function loadPickledDF(filename::AbstractString;
                       migrate::Bool=true)
    f = pyeval("open(\"$filename\", \"rb\")")
    @pyimport pickle
    pydf = pickle.load(f)
    df = convertPyDF(pydf, migrate=migrate)
    return df
end
export loadPickledDF


"""
Shuffles a dataframe in place.
"""
function shuffle!(df::DataFrame)
    permutation = shuffle(collect(1:size(df)[1]))
    tdf = copy(df)
    for i in 1:length(permutation)
        df[i, :] = tdf[permutation[i], :]
    end
    return df
end
export shuffle!


"""
Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array
so it always maps to an integer
"""
function numericalCategories(otype::DataType, A::Array)
    mapping = sort!(unique(A))
    o = convert(Array{otype}, indexin(A, mapping))
    return o, mapping
end
# define for DataArray type
numericalCategories(otype::DataType, A::DataArray) = numericalCategories(otype, 
        convert(Array, A))
export numericalCategories


"""
Gets the default mapping of categorical variables which would be returned by
numericalCategories.
"""
function getDefaultCategoricalMapping(A::Array)
    return sort!(unique(A))
end
export getDefaultCategoricalMapping


"""
Converts a categorical value into numerical values of the given type.

Returns the mapping.
"""
function numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)
    df[symbol(string(col)*"_Orig")] = df[col]
    df[col], mapping = numericalCategories(otype, df[col])
    return mapping
end
export numericalCategories!


"""
Converts categorical variables into numerical values for multiple columns in a
dataframe.  For now doesn't return mapping, may have to implement some type of 
mapping type.
"""
function numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol})
    for col in cols
        numericalCategories!(otype, df, col)
    end
    return
end
export numericalCategories!


"""
Converts all null values (NaN's and Nullable()) to a particular value.
Note this has to check whether the type is Nullable.
"""
function convertNulls!{T}(A::Array{T, 1}, newvalue::T)
    if T <: Nullable
        for i in 1:length(A)
            if isnull(A[i])
                A[i] = newvalue
            end
        end
    end
    if T <: AbstractFloat
        for i in 1:length(A)
            if isnan(A[i])
                A[i] = newvalue
            end
        end
    end
    return A
end
export convertNulls!


"""
Converts all null vlaues (NA's, NaN's and Nullable()) to a particular value.
"""
function convertNulls{T}(A::DataArray{T}, newvalue::T)
    A = convert(Array, A, newvalue)
    convertNulls!(A, newvalue)
    return DataArray(A)
end
export convertNulls


"""
Convert all null values in columns of a DataFrame to a particular value.
"""
function convertNulls!(df::DataFrame, cols::Array{Symbol, 1}, newvalue::Any)
    for col in cols
        df[col] = convertNulls(df[col], newvalue)
    end
    return
end
convertNulls!(df::DataFrame, col::Symbol, newvalue) = convertNulls!(df, [col], newvalue)
export convertNulls!


"""
The default copy method for dataframe only copies one level deep, so basically it stores
an array of columns.  If you assign elements of individual (column) arrays then, it can
make changes to references to those arrays that exist elsewhere.

This method instead creates a new dataframe out of copies of the (column) arrays.

This is not named copy due to the fact that there is already an explicit copy(::DataFrame)
implementation in dataframes.
"""
function copyColumns(df::DataFrame)
    ndf = DataFrame()
    for col in names(df)
        ndf[col] = copy(df[col])
    end
    return ndf
end
export copyColumns


