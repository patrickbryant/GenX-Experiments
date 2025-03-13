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
using Statistics
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


function run_case(case="three_zones")
    include(joinpath(case, "Run.jl"))
    return case
end

function workhorse_function(case="three_zones_cheap_storage")
    #
    # Play with results (Time Domain Reduction)
    #
    Period_map = CSV.read(joinpath(case,"TDR_Results/Period_map.csv"),DataFrame,missingstring="NA")
    # get demand data
    demand_tdr = CSV.read(joinpath(case,"TDR_results/Demand_data.csv"),DataFrame,missingstring="NA")[!,["Demand_MW_z1","Demand_MW_z2","Demand_MW_z3"]]
    demand_tdr.Demand_MW = demand_tdr.Demand_MW_z1 + demand_tdr.Demand_MW_z2 + demand_tdr.Demand_MW_z3
    demand = CSV.read(joinpath(case,"system/Demand_data.csv"),DataFrame,missingstring="NA")[!,["Time_Index","Demand_MW_z1","Demand_MW_z2","Demand_MW_z3"]]
    demand.Demand_MW = demand.Demand_MW_z1 + demand.Demand_MW_z2 + demand.Demand_MW_z3

    # Find array of unique representative periods
    rep_periods = unique(Period_map[!,"Rep_Period"])

    # Create an array of the time steps and MW values of each representative period
    demand_rep = reduce(vcat, [[repeat([i],168) demand[(168*i-167):168*i,"Time_Index"] demand[(168*i-167):168*i,"Demand_MW"]] for i in rep_periods])

    # Combine with Total (pre TDR)
    demand_plot = [repeat(["Total"],8760) demand.Time_Index demand.Demand_MW];

    # Add column names and convert column type
    demand_tdr_plot = [demand_plot; demand_rep]
    demand_tdr_plot = DataFrame(demand_tdr_plot, ["Week","Hour", "MW"])
    demand_tdr_plot.Hour = convert.(Int64,   demand_tdr_plot.Hour);
    demand_tdr_plot.MW   = convert.(Float64, demand_tdr_plot.MW);
    
    p=demand_tdr_plot |>
        @vlplot(mark={:line},
                x={:Hour,title="Time Step (hours)",labels="Week:n"}, y={:MW,title="Demand (MW)"},
                color={"Week:n", scale={scheme="paired"},sort="decsending"}, title="MW Demand per hour with TDR Representative Weeks",
                width=845,height=400)
    p |> save(joinpath(case,"TDR_Results/demand_tdr.pdf"))

    # Reconstruct annual demand from representative periods
    demand_rec = reduce(vcat, [[repeat([r],168) collect((168*i-167):168*i) demand[(168*r-167):168*r,"Demand_MW"]] for (i,r) in zip(Period_map.Period_Index, Period_map.Rep_Period)])
    demand_rec = DataFrame(demand_rec, ["Week","Hour", "MW"])
    demand_rec.Hour = convert.(Int64,   demand_rec.Hour);
    demand_rec.MW   = convert.(Float64, demand_rec.MW);

    p=demand_rec |>
        @vlplot(mark={:line},
                x={:Hour,title="Time Step (hours)",labels="Week:n"}, y={:MW,title="Demand (MW)"},
                color={"Week:n", scale={scheme="paired"},sort="decsending"}, title="Reconstructed MW Demand per hour from TDR Representative Weeks",
                width=845,height=400)
    p |> save(joinpath(case,"TDR_Results/demand_rec.pdf"))

    #Get daily average, ie MW-day or MWd    
    demand.Demand_MWd = vec(repeat(mean(reshape(demand.Demand_MW,24,365), dims=1), 24))
    demand_rec.   MWd = vec(repeat(mean(reshape(demand_rec.   MW,24,364), dims=1), 24))#; demand.Demand_MW[8737:end]]
    demand_daily_average = DataFrame([repeat(demand.Time_Index, 2) [demand.Demand_MWd; demand_rec.MWd; repeat([0.],24)] [repeat(["True"], 8760); repeat(["Reconstructed"], 8760)]], ["Hour", "MW", "Source"])
    p=demand_daily_average |>
        @vlplot(mark={:line},
                x={:Hour,title="Time Step (hours)",labels="Source:n", type=:quantitative}, y={:MW,title="Demand (MW)", type=:quantitative},
                color={"Source:n", scale={scheme="accent"},sort="decsending"},
                title="Daily Average MW Demand",
                width=845,height=400)
    p |> save(joinpath(case,"TDR_Results/demand_daily_average.pdf"))
    
    #Get weekly average, ie MW-week or MWw
    demand.Demand_MWw = [vec(repeat(mean(reshape(demand.Demand_MW[1:8736],168,52), dims=1), 168)); demand.Demand_MW[8737:end]]
    demand_rec.   MWw =  vec(repeat(mean(reshape(demand_rec.   MW[1:8736],168,52), dims=1), 168))
    demand_weekly_average = DataFrame([repeat(demand.Time_Index, 2) [demand.Demand_MWw; demand_rec.MWw; repeat([0.],24)] [repeat(["True"], 8760); repeat(["Reconstructed"], 8760)]], ["Hour", "MW", "Source"])
    p=demand_weekly_average |>
        @vlplot(mark={:line},
                x={:Hour,title="Time Step (hours)",labels="Source:n", type=:quantitative}, y={:MW,title="Demand (MW)", type=:quantitative},
                color={"Source:n", scale={scheme="accent"},sort="decsending"},
                title="Weekly Average MW Demand",
                width=845,height=400)
    p |> save(joinpath(case,"TDR_Results/demand_weekly_average.pdf"))

    
    #
    # Results
    #

    #results = "results_CO2_cap_1_same_emissions_as_nominal"
    results = "results_1"
    power    = CSV.read(joinpath(case,results,"power.csv"),       DataFrame,missingstring="NA")[3:end,:] # time series starts at index 3
    charge   = CSV.read(joinpath(case,results,"charge.csv" ),     DataFrame,missingstring="NA")[3:end,:] # time series starts at index 3
    storage  = CSV.read(joinpath(case,results,"storage.csv"),     DataFrame,missingstring="NA")[3:end,:] # time series starts at index 3
    curtail  = CSV.read(joinpath(case,results,"curtailment.csv"), DataFrame,missingstring="NA")[3:end,:] # time series starts at index 3
    capacity = CSV.read(joinpath(case,results,"capacity.csv"),    DataFrame,missingstring="NA")
    flow     = CSV.read(joinpath(case,results,"flow.csv"),        DataFrame,missingstring="NA")
    # Pre-processing
    tstart = 1#36
    tend   = 240#60 
    istart = tstart+0#removed header rows so first index is first time index
    iend   = tend  +0
    N = tend-tstart+1
    hour = collect(tstart:tend)
    # names_power = ["Solar","Natural_Gas","Battery","Wind"]
    names_power = ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"]

    # add up total power over zones. Subtract battery charging to give net battery output. Area plots freak out if you have zeros in the mostly negative charging curve. Add a small negative constant to fix numerical issue in plot. 
    power_tot = DataFrame([power[!,9]+power[!,10]+power[!,11] -0.001.-(charge[!,2]+charge[!,3]+charge[!,4]) power[!,5]+power[!,7] power[!,6]+power[!,8] power[!,2]+power[!,3]+power[!,4]],
                          ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"])

    power_z1  = DataFrame([power[!,9] -0.001.-(charge[!,2]) power[!,5] zero(power[!,6]) power[!,2]],    # there's no wind generation in zone 1
                          ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"])
    power_z2  = DataFrame([power[!,10] -0.001.-(charge[!,3]) power[!,7] power[!,6] power[!,3]],
                          ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"])
    power_z3  = DataFrame([power[!,11] -0.001.-(charge[!,4]) zero(power[!,7]) power[!,8] power[!,4]],  # there's no solar generation in zone 3
                          ["Battery_Discharge","Battery_Charge","Solar","Wind","Natural_Gas"])

    # make plot dataframe grouped by resource type
    power_plot = DataFrame([hour power_tot[istart:iend,1] repeat([names_power[1]],N)],
                           ["Hour","MW", "Resource_Type"])

    power_pz1  = DataFrame([hour power_z1[istart:iend,1] repeat([names_power[1]],N)],
                           ["Hour","MW", "Resource_Type"])
    power_pz2  = DataFrame([hour power_z2[istart:iend,1] repeat([names_power[1]],N)],
                           ["Hour","MW", "Resource_Type"])
    power_pz3  = DataFrame([hour power_z3[istart:iend,1] repeat([names_power[1]],N)],
                           ["Hour","MW", "Resource_Type"])
    
    for i in 2:length(names_power)
        power_plot_temp = DataFrame([hour power_tot[istart:iend,i] repeat([names_power[i]],N)],
                                    ["Hour","MW", "Resource_Type"])
        power_plot = [power_plot; power_plot_temp]

        power_pz1  = [power_pz1; DataFrame([hour power_z1[istart:iend,i] repeat([names_power[i]],N)],
                                           ["Hour","MW", "Resource_Type"])]
        power_pz2  = [power_pz2; DataFrame([hour power_z2[istart:iend,i] repeat([names_power[i]],N)],
                                           ["Hour","MW", "Resource_Type"])]
        power_pz3  = [power_pz3; DataFrame([hour power_z3[istart:iend,i] repeat([names_power[i]],N)],
                                           ["Hour","MW", "Resource_Type"])]
    end

    # add transmission data
    power_plot.Transmission12 = repeat(flow[istart:iend,"1"], length(names_power))
    power_plot.Transmission13 = repeat(flow[istart:iend,"2"], length(names_power))

    power_pz1.Transmission12 = repeat(flow[istart:iend,"1"], length(names_power))
    power_pz1.Transmission13 = repeat(flow[istart:iend,"2"], length(names_power))
    power_pz2.Transmission12 = repeat(flow[istart:iend,"1"], length(names_power))
    power_pz3.Transmission13 = repeat(flow[istart:iend,"2"], length(names_power))
    
    demand_tdr_tot = demand_tdr.Demand_MW_z1+demand_tdr.Demand_MW_z2+demand_tdr.Demand_MW_z3
    power_plot.Demand_Total = repeat(demand_tdr_tot[istart:iend], length(names_power))
    power_pz1.Demand_Total = repeat(demand_tdr[istart:iend,"Demand_MW_z1"], length(names_power))
    power_pz2.Demand_Total = repeat(demand_tdr[istart:iend,"Demand_MW_z2"], length(names_power))
    power_pz3.Demand_Total = repeat(demand_tdr[istart:iend,"Demand_MW_z3"], length(names_power))

    # Compute net generation for comparison with demand
    net_generation = power_tot[!,1]+power_tot[!,2]+power_tot[!,3]+power_tot[!,4]+power_tot[!,5]
    power_plot.Net_Generation = repeat(net_generation[istart:iend], length(names_power))
    power_pz1.Net_Generation = repeat(power_z1[istart:iend,1]+power_z1[istart:iend,2]+power_z1[istart:iend,3]+power_z1[istart:iend,4]+power_z1[istart:iend,5], length(names_power))
    power_pz1.Net_Generation = power_pz1.Net_Generation - power_pz1.Transmission12 - power_pz1.Transmission13
    power_pz2.Net_Generation = repeat(power_z2[istart:iend,1]+power_z2[istart:iend,2]+power_z2[istart:iend,3]+power_z2[istart:iend,4]+power_z2[istart:iend,5], length(names_power))
    power_pz2.Net_Generation = power_pz2.Net_Generation + power_pz2.Transmission12
    power_pz3.Net_Generation = repeat(power_z3[istart:iend,1]+power_z3[istart:iend,2]+power_z3[istart:iend,3]+power_z3[istart:iend,4]+power_z3[istart:iend,5], length(names_power))
    power_pz3.Net_Generation = power_pz3.Net_Generation + power_pz3.Transmission13

    p = power_plot |>
        @vlplot()+
        @vlplot(mark={:area},
                x={:Hour,title="Time Step (hours)",labels="Resource_Type:n",axis={values=tstart:12:tend},type=:quantitative},
                y={:MW,title="Load (MW)",type=:quantitative},
                color={"Resource_Type:n",scale={scheme="accent"},sort="descending"},order={field="Resource_Type:n"},
                width=845,height=400,
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Demand_Total, type=:quantitative},
                lables="Demand",color={datum="Demand",legend={title=nothing}},title="Resource Capacity per Hour with Load Demand Curve, all Zones",
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Net_Generation, type=:quantitative},
                labels="Net_Generation",color={datum="Net_Generation",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission12, type=:quantitative},
                labels="Transmission12",color={datum="Transmission12",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission13, type=:quantitative},
                labels="Transmission13",color={datum="Transmission13",legend={title=nothing}},
                )
    p |> save(joinpath(case,results,"power.pdf"))

    p = power_pz1 |>
        @vlplot()+
        @vlplot(mark={:area},
                x={:Hour,title="Time Step (hours)",labels="Resource_Type:n",axis={values=tstart:12:tend},type=:quantitative},
                y={:MW,title="Load (MW)",type=:quantitative},
                color={"Resource_Type:n",scale={scheme="accent"},sort="descending"},order={field="Resource_Type:n"},
                width=845,height=400,
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Demand_Total, type=:quantitative},
                lables="Demand",color={datum="Demand",legend={title=nothing}},title="Resource Capacity per Hour with Load Demand Curve, Zone 1",
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Net_Generation, type=:quantitative},
                labels="Net_Generation",color={datum="Net_Generation",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission12, type=:quantitative},
                labels="Transmission12",color={datum="Transmission12",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission13, type=:quantitative},
                labels="Transmission13",color={datum="Transmission13",legend={title=nothing}},
                )
    p |> save(joinpath(case,results,"power_z1.pdf"))

    p = power_pz2 |>
        @vlplot()+
        @vlplot(mark={:area},
                x={:Hour,title="Time Step (hours)",labels="Resource_Type:n",axis={values=tstart:12:tend},type=:quantitative},
                y={:MW,title="Load (MW)",type=:quantitative},
                color={"Resource_Type:n",scale={scheme="accent"},sort="descending"},order={field="Resource_Type:n"},
                width=845,height=400,
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Demand_Total, type=:quantitative},
                lables="Demand",color={datum="Demand",legend={title=nothing}},title="Resource Capacity per Hour with Load Demand Curve, Zone 2",
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Net_Generation, type=:quantitative},
                labels="Net_Generation",color={datum="Net_Generation",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission12, type=:quantitative},
                labels="Transmission12",color={datum="Transmission12",legend={title=nothing}},
                )
    p |> save(joinpath(case,results,"power_z2.pdf"))

    p = power_pz3 |>
        @vlplot()+
        @vlplot(mark={:area},
                x={:Hour,title="Time Step (hours)",labels="Resource_Type:n",axis={values=tstart:12:tend},type=:quantitative},
                y={:MW,title="Load (MW)",type=:quantitative},
                color={"Resource_Type:n",scale={scheme="accent"},sort="descending"},order={field="Resource_Type:n"},
                width=845,height=400,
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Demand_Total, type=:quantitative},
                lables="Demand",color={datum="Demand",legend={title=nothing}},title="Resource Capacity per Hour with Load Demand Curve, Zone 3",
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Net_Generation, type=:quantitative},
                labels="Net_Generation",color={datum="Net_Generation",legend={title=nothing}},
                )+
        @vlplot(mark=:line,
                x={:Hour, type=:quantitative},
                y={:Transmission13, type=:quantitative},
                labels="Transmission13",color={datum="Transmission13",legend={title=nothing}},
                )
    p |> save(joinpath(case,results,"power_z3.pdf"))
    

    storage_plot = DataFrame( [hour.+0.5 storage[istart:iend, 2] repeat([capacity[8,"EndEnergyCap"]], N) hour power[istart:iend, 9]-charge[istart:iend, 2] repeat([capacity[8,"EndCap"]], N) min.(power[istart:iend, 9], charge[istart:iend, 2]) repeat(["1"], N)],
                              ["Hour_storage", "MWh", "MWh_max", "Hour", "MW", "MW_max", "MW_circ", "Zone"] )
    for z in range(2,3)
        storage_plot_temp = DataFrame( [hour.+0.5 storage[istart:iend, 1+z] repeat([capacity[7+z,"EndEnergyCap"]], N) hour power[istart:iend, 8+z]-charge[istart:iend, 1+z] repeat([capacity[7+z,"EndCap"]], N) min.(power[istart:iend, 8+z], charge[istart:iend, 1+z]) repeat([string(z)], N)],
                                       ["Hour_storage", "MWh", "MWh_max", "Hour", "MW", "MW_max", "MW_circ", "Zone"] )
        storage_plot = [storage_plot; storage_plot_temp]
    end
    storage_plot.MW_curtail = [curtail[istart:iend, 5];
                               curtail[istart:iend, 6]+curtail[istart:iend, 7];
                               curtail[istart:iend, 8]]


    p = storage_plot |> @vlplot() +
        [
            @vlplot(layer=[{mark={:line, clip=true},
                            x={field=:Hour_storage, title=nothing, type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={field=:MWh, title="Stored Energy (MWh)", type=:quantitative,
                               # scale={domain=(10000,10200)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            width=845, height=400,
                            },
                           {mark={:line, clip=true, strokeDash=[8,8]},
                            x={field=:Hour_storage, title=nothing, type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={:MWh_max, type=:quantitative,
                               # scale={domain=(10000,10200)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            }]) 
            @vlplot(layer=[{mark={:point, filled=true, clip=true},
                            x={field=:Hour, title="Time Step (hours)", type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={field=:MW, title="Net Discharge Power (MW)", type=:quantitative,
                               # scale={domain=(-200,-150)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            width=845, height=400,
                            },
                           {mark={:line, clip=true, strokeDash=[8,8]},
                            x={field=:Hour, title=nothing, type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={field="MW_max", type=:quantitative,
                               # scale={domain=(-200,-150)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            }])
            @vlplot(layer=[{mark={:point, filled=true, clip=true},
                            x={field=:Hour, title="Time Step (hours)", type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={field=:MW_circ, title="Simultaneous Charge/Discharge Power (MW)", type=:quantitative,
                               # scale={domain=(800,1200)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            width=845, height=400,
                            },
                           {mark={:line, strokeDash=[8,8], clip=true},
                            transform=[{calculate="datum.MW_max/2", as="half_MW_max"}],
                            x={field=:Hour, title=nothing, type=:quantitative,
                               labels="Zone:n",
                               axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
                            y={field="half_MW_max", type=:quantitative,
                               # scale={domain=(800,1200)},
                               },
                            color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
                            }])
            # @vlplot(mark={:point, filled=true},
            #         x={field=:Hour, title="Time Step (hours)", type=:quantitative,
            #            labels="Zone:n",
            #            axis={values=tstart:12:tend+1}, scale={domain=[tstart,tend+1]}},
            #         y={field=:MW_curtail, title="Curtailed Power (MW)", type=:quantitative},
            #         color={"Zone:n",scale={scheme="accent"},sort="ascending"},order={field="Zone:n"},
            #         width=845, height=400,
            #         )
        ]
    
    p |> save(joinpath(case,results,"storage.pdf"))
    
end


function main()
   workhorse_function()
end

!isinteractive() && main()
