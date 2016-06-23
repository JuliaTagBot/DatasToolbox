
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
    for col in names(df)
        isinstance = pybuiltin("isinstance")
        pystr = pybuiltin("str")
        if isinstance(df[1, col], pystr)
            df[col] = convertCol(df, col, ASCIIString)
        elseif isinstance(df[1, col], datetime.date)
            df[col] = convertCol(df, col, Date)
        elseif isinstance(df[1, col], datetime.datetime)
            df[col] = convertCol(df, col, DateTime)
        end
    end
end
        

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

