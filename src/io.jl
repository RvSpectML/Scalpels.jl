#using LinearAlgebra    # For dot and norm (calc_cos_similarity, etc.)
#using Statistics       # For mean (MultivariateStats)
using FITSIO         # For FITS file reading

"""
    make_list_of_filename(dir, pattern)

Generate a vector of filenames within directory `dir` that match the regular expression `pattern`.
"""
function make_list_of_filename(dir, pattern)
    file_list = String[]
    for (root, dirs, files) in walkdir(dir)
        for file in files
            if occursin(pattern, file)
                push!(file_list, joinpath(root, file))
            end
        end
    end
    return file_list
end

"""
    read_data_from_fits(filename, hdu_name)

Open FITS file and read the specified HDU.
"""
function read_data_from_fits(filename, hdu_name)
    fits_file = FITS(filename)
    data = read(fits_file[hdu_name])
    close(fits_file)
    return data
end

"""
    read_data_from_fits(filename, hdu_names::Vector{String})

Open FITS file, read each of the the specified HDUs and return them as a named tuple.
"""
function read_data_from_fits(filename, hdu_names::Vector{String})
    fits_file = FITS(filename)
    data_dict = Dict{Symbol, Any}()
    for hdu_name in hdu_names
        data_dict[Symbol(hdu_name)] = read(fits_file[hdu_name])
    end
    close(fits_file)
    return (; data_dict...)
end

"""
    read_data_from_filelist(filenames, hdu_names)

Inputs:
- `filenames`: List of FITS files.
- `hdunames`: List of HDU names to read from FITS files.

Output:
A dictionary with keys from hdu_names and values being Matrices.
The columns of each matrix contains data from one file.
"""
function read_data_from_filelist(filenames, hdu_names)
    # Determine the size of the data for preallocation
    local_data = read_data_from_fits(filenames[1], hdu_names)
    output_data = Dict{Symbol, Matrix}()

    for hdu_name in hdu_names
        # Note: Size is determined from the first file. Assumes consistent data shape.
        output_data[Symbol(hdu_name)] = zeros(eltype(local_data[Symbol(hdu_name)]), size(local_data[Symbol(hdu_name)])..., length(filenames))
    end

    for (idx, filename) in enumerate(filenames)
        local_data = read_data_from_fits(filename, hdu_names)
        for hdu_name in hdu_names
            # Assuming data is a vector or matrix where the last index is the observation index
            if ndims(local_data[Symbol(hdu_name)]) == 1
                 output_data[Symbol(hdu_name)][:, idx] = local_data[Symbol(hdu_name)]
            else
                 # Simple assignment for 2D data (e.g., CCF vs time)
                 # This part requires more context on data dimension, but based on the original snippet:
                 # output_data[:CCF][:, idx] = local_data[:CCF]
                 @warn "Multi-dimensional data in FITS HDU $(hdu_name). Assuming 1D-per-file."
            end
        end
    end

    # Correction based on the original code logic which used [:, idx] assignment:
    for (idx, filename) in enumerate(filenames)
        local_data = read_data_from_fits(filename, hdu_names)
        for hdu_name in hdu_names
            output_data[Symbol(hdu_name)][:, idx] = local_data[Symbol(hdu_name)]
        end
    end

    return output_data
end

"""
    read_data_from_multiple_instruments(dir, pattern_list, hdu_names)

Inputs:
- `dir`: Directory containing input files.
- `pattern_list`: A list of regular expressions for selecting filenames.
- `hdu_list`: A list of HDUs to read from FITS files.

Output:
A vector of Dictionaries, one for each pattern in pattern_list.
Each Dictionary contains keys from hdu_names and values read from the input files.
"""
function read_data_from_multiple_instruments(dir, pattern_list, hdu_names)
    all_instrument_data = fill(Dict{Symbol,AbstractMatrix}(),length(pattern_list) )

    for (i,pattern) in enumerate(pattern_list)
        filenames = make_list_of_filename(dir, pattern)
        if isempty(filenames)
            @warn "No files found for pattern: $pattern in directory: $dir"
            continue
        end

        instrument_data = read_data_from_filelist(filenames, hdu_names)
        all_instrument_data[i] = instrument_data
    end

    # Filter out empty dictionaries
    return filter(d -> !isempty(d), all_instrument_data)
end
