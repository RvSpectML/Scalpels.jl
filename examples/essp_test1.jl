using Scalpels

inst_name = ["neid", "expres", "harps", "harpsn"]

data_dir = "data/DS1/CCFs/"
filenames_for_inst = map(i->make_list_of_filename(data_dir, Regex("^DS\\d+\\.\\d+_ccfs_" * inst_name[i] * "\\.fits\$") ), 1:length(inst_name) )

hdus_to_read = ["CCF","E_CCF"]
data_inst1 = read_data_from_filelist(filenames_for_inst[1], hdus_to_read )


patterns_for_inst = map(i->Regex("^DS\\d+\\.\\d+_ccfs_" * inst_name[i] * "\\.fits\$"), 1:length(inst_name))
data = read_data_from_multiple_instruments(data_dir, patterns_for_inst, hdus_to_read)

#results = map(dat->init_guess_linear_model(dat[:CCF]), data )

#=
# using Plots
"Function to plot the mean vectors in data"
function plot_mean_vectors(data)
    plot()
    for i in 1:size(data.mean_spectrum, 2)
        plot!(data.mean_spectrum[:, i], label="Mean Spectrum $(i)")
    end
    display(plot!())
end
=#
