
"""
Type for handling time series data.  As with DataHandler it is intended taht most of
the reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.
"""
type TimeSeriesHandler{T} <: AbstractDH{T}

    # dataframe where all data is kept
    df::DataFrame

    colsInput::Array{Symbol, 1}
    colsOutput::Array{Symbol, 1}

    colsNormalize::Array{Symbol, 1}

    mu::Array{T, 1}
    norm::Array{T, 1}
    userange::Bool

    # training data, is a subset of df
    dfTrain::DataFrame
    # testing data, is a subset of df
    dfTest::DataFrame

    # these are the arrays of training, testing data
    X_train::Array{T}
    y_train::Array{T}
    X_test::Array{T}
    y_test::Array{T}

    # this is where predictions are kept
    yhat::Array{T}
    yhat_train::Array{T}

    # the symbol of the timeindex
    timeindex::Symbol
    # the length of sequences (only fixed for now)
    seq_length::Int32

    """
    This constructor makes a fractional train, test split.  
    """
    function DataHandler(df::DataFrame, timeindex::Symbol, seq_length::Integer; 
                         testfrac::AbstractFloat=0., shuffle::Bool=false,
                         input_cols::Array{Symbol}=Symbol[], 
                         output_cols::Array{Symbol}=Symbol[],
                         normalize_cols::Array{Symbol}=Symbol[],
                         assign::Bool=false,
                         userange::Bool=false)
        ndf = copy(df)
        o = new(ndf, input_cols, output_cols, normalize_cols)
        o.userange = userange
        o.timeindex = timeindex
        split!(o, testfrac, shuffle=shuffle, assign=assign)
        computeNormalizeParameters!(o, dataset=:dfTrain)
        if canNormalize(o)
            normalizeTrain!(o)
            if size(o.dfTest)[1] > 0 normalizeTest!(o) end
        end
        return o
    end
end
export TimeSeriesHandler


"""
Assigns the training data.  X output will be of shape (samples, seq_length, seq_width).
One should be extremely careful if not sorting.

Note that this is silent if the dataframe is empty.
"""
function assignTrain!{T}(dh::TimeSeriesHandler{T}; sort::Bool=true)
    if isempty(dh.dfTrain) return end
    if sort sort!(dh.dfTrain, cols=[timeindex]) end 
    X_unshaped = convert(Array{T}, dh.dfTrain[:, dh.colsInput])
    # TODO this was copy pasted, not from here
    npoints = size(tdf)[1] - seq_length
    X = zeros(npoints, seq_length, length(cols))
    y = zeros(npoints, 1)
    # the loop can only go this far
    for i in 1:npoints
        # nextx = [convert(Array, tdf[i+j, cols]) for j in 0:(seq_length-1)]
        nextx = reshape(convert(Array, tdf[i:(i+seq_length-1), cols]),
                        (1, seq_length, length(cols)))
        X[i, :, :] = nextx
        y[i, 1] = tdf[i+seq_length, cols[1]]
    end
    p = 0
    if shuffle
        X, y = shuffle_data(X, y)
    end
    return X, y
end

