#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve() #Optional if a known-good Manifest.toml is included
Pkg.instantiate()
# Pkg.add("ImageMagick")
# Pkg.add("PrettyPrint")
# Pkg.add("CSV")
# Pkg.add("YAML")
# Pkg.add("GraphRecipes")
# Pkg.add("VegaLite")
# Pkg.add("StatsPlots")

using PrettyPrint
using DataFrames
using CSV
using YAML
using GraphRecipes
using Plots
using PlotlyJS
using VegaLite
using StatsPlots
using GenX


function data_loader(case="three_zones")
    # load & preprocess your data here

    #
    # Setup the run
    #
    time_domain_reduction_settings = YAML.load(open(joinpath(case,"settings/time_domain_reduction_settings.yml")))
    pprintln(time_domain_reduction_settings)

    #
    # Run the case
    #

    # include("three_zones/Run.jl")
    return case
end


function workhorse_function(case)
   # where the main data munging/process happens

    #
    # Play with results
    #
    results = cd(readdir,joinpath(case,"results"))

    power   = CSV.read(joinpath(case,"results/power.csv"),DataFrame,missingstring="NA")
    charge  = CSV.read(joinpath(case,"results/charge.csv" ),DataFrame,missingstring="NA")
    storage = CSV.read(joinpath(case,"results/storage.csv"),DataFrame,missingstring="NA")
    # Pre-processing
    tstart = 4 #0+3
    tend   = 120 #7*24+2
    istart = tstart+3
    iend   = tend  +3
    N = tend-tstart+1
    hour = collect(tstart:tend)
    # names_power = ["Solar","Natural_Gas","Battery","Wind"]
    names_power = ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"]

    # add up total power over zones. Subtract battery charging to give net battery output. Area plots freak out if you have zeros in the mostly negative charging curve. Add a small negative constant to fix numerical issue in plot. 
    power_tot = DataFrame([power[!,9]+power[!,10]+power[!,11] -0.001.-(charge[!,9]+charge[!,10]+charge[!,11]) power[!,5]+power[!,7] power[!,6]+power[!,8] power[!,2]+power[!,3]+power[!,4]],
                          ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"])

    # power_tot = DataFrame([power[!,5]+power[!,7] power[!,2]+power[!,3]+power[!,4] power[!,9]-charge[!,9]+power[!,10]-charge[!,10]+power[!,11]-charge[!,11] power[!,6]+power[!,8]],
    #                       ["Solar","Natural_Gas","Battery","Wind"])
    
    # make plot dataframe grouped by resource type
    power_plot = DataFrame([hour power_tot[istart:iend,1] repeat([names_power[1]],N)],
                           ["Hour","MW", "Resource_Type"]);

    for i in range(2,5)
        power_plot_temp = DataFrame([hour power_tot[istart:iend,i] repeat([names_power[i]],N)],
                                    ["Hour","MW", "Resource_Type"])
        power_plot = [power_plot; power_plot_temp]
    end

    # get demand data, should think about plotting from TDR...
    loads =  CSV.read(joinpath(case,"TDR_results/Demand_data.csv"),DataFrame,missingstring="NA")
    loads_tot = loads[!,"Demand_MW_z1"]+loads[!,"Demand_MW_z2"]+loads[!,"Demand_MW_z3"]
    power_plot[!,"Demand_Total"] = repeat(loads_tot[istart-2:iend-2],5); # two fewer header rows in demand_data.csv!!

    # Compute net generation for comparison with demand
    net_generation = power_tot[!,1]+power_tot[!,2]+power_tot[!,3]+power_tot[!,4]+power_tot[!,5]
    power_plot[!,"Net_Generation"] = repeat(net_generation[istart:iend],5)

    p = power_plot |>
        @vlplot()+
        @vlplot(mark={:area},
                x={:Hour,title="Time Step (hours)",labels="Resource_Type:n",axis={values=tstart:12:tend}},
                y={:MW,title="Load (MW)",type="quantitative"},
                color={"Resource_Type:n",scale={scheme="accent"},sort="descending"},order={field="Resource_Type:n"},
                width=845,height=400)+
                    @vlplot(mark=:line,
                            x=:Hour,
                            y=:Demand_Total,
                            lables="Demand",color={datum="Demand",legend={title=nothing}},title="Resource Capacity per Hour with Load Demand Curve, all Zones")+
                                @vlplot(mark=:line,
                                        x=:Hour,
                                        y=:Net_Generation,
                                        labels="Net_Generation",color={datum="Net_Generation",legend={title=nothing}})
    
    

    p |> save("test/power.pdf")

    storage_plot = DataFrame( [hour.+0.5 storage[istart-1:iend-1, 9] hour power[istart:iend, 9]-charge[istart:iend, 9] repeat(["1"], N)],
                              ["Hour_storage", "MWh", "Hour", "MW", "Zone"] )
    for z in range(2,3)
        storage_plot_temp = DataFrame( [hour.+0.5 storage[istart-1:iend-1, 8+z] hour power[istart:iend, 8+z]-charge[istart:iend, 8+z] repeat([string(z)], N)],
                                       ["Hour_storage", "MWh", "Hour", "MW", "Zone"] )
        storage_plot = [storage_plot; storage_plot_temp]
    end

    p = storage_plot |>
        @vlplot()+
        [@vlplot(mark=:line,
                 x={field=:Hour_storage, title=nothing,
                    labels="Zone:n",
                    axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}, type=:quantitative},
                 y={field=:MWh, title="Stored Energy (MWh)", type=:quantitative},
                 color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                 width=845, height=400,
                 )
         @vlplot(mark=:point,
                 x={field=:Hour, title="Time Step (hours)",
                    labels="Zone:n",
                    axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}, type=:quantitative},
                 y={field=:MW, title="Net Discharge Power (MW)", type=:quantitative},
                 color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                 width=845, height=400,
                 )]
    
    p |> save("test/storage.pdf")
    
end


function main()
   workhorse_function(data_loader())
end

!isinteractive() && main()
