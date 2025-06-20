import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colormaps
from matplotlib.collections import PatchCollection

# https://stackoverflow.com/questions/31908982/multi-color-legend-entry
# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch
    
def colored_line(x, y, c, ax, **lc_kwargs): # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt",
                      "colors": c}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    # lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


names_power = ["Hydro",   "Nuclear",    "Coal", "Natural Gas",   "Solar",    "Wind"]#, "Curtailed Wind", "Curtailed Solar", "Generation",  "Demand"]
color_power = ["#3D7D92", "#76150C", "#222222",     "#4C2469", "#F7CB46", "#52B3EA"]#,        "#AFD5EA",         "#F7E8B9",    "#FF0000", "#0000FF"]

case_TDR_demand = 'three_zones_100hr_TDR_demand'
case_algorithms = 'three_zones_100hr'

period_map = pd.read_csv(f'{case_algorithms}/TDR_Results/Period_map.csv')

demand_raw = pd.read_csv(f'{case_algorithms}/system/Demand_data.csv')[['Time_Index','Demand_MW_z1','Demand_MW_z2','Demand_MW_z3']]
demand_TDR = pd.read_csv(f'{case_TDR_demand}/system/Demand_data.csv')[['Time_Index','Demand_MW_z1','Demand_MW_z2','Demand_MW_z3']]
demand_raw['Demand_MW'] = demand_raw[['Demand_MW_z1','Demand_MW_z2','Demand_MW_z3']].sum(axis=1)
demand_TDR['Demand_MW'] = demand_TDR[['Demand_MW_z1','Demand_MW_z2','Demand_MW_z3']].sum(axis=1)

demand_raw['Week'] = (demand_raw.Time_Index-1)//168+1
demand_raw['In_Rep_Period'] = np.zeros(8760)
demand_raw.loc[pd.Index(period_map.Rep_Period.unique()).get_indexer(demand_raw.Week) >= 0, 'In_Rep_Period'] = 1

demand_TDR['Rep_Period'] = np.array(period_map.Rep_Period.repeat(168))

# Illustrate replacement of 8760 demand with 8736 reconstructed from TDR
fig = plt.figure(figsize=(8,4), layout='tight')
plt.xlabel('Hour')
plt.ylabel('Demand [GW]')
plt.plot(demand_raw.Time_Index, demand_raw.Demand_MW/1000, label='8760')
ax = plt.gca()
colored_line(demand_raw.Time_Index, demand_raw.Demand_MW/1000, colormaps['rainbow'](demand_raw.Week/52)*np.array(demand_raw.In_Rep_Period).reshape(-1,1), ax)
h, l = ax.get_legend_handles_labels()
h.append(MulticolorPatch(colormaps['rainbow'](np.linspace(0,1,11))))
l.append('Representative Weeks')
ax.set_xlim([1,8760])
ax.set_ylim([0,25])
plt.legend(h,l, loc='best', handler_map={MulticolorPatch: MulticolorPatchHandler()})
fig.savefig('plots/demand_raw.pdf')

fig = plt.figure(figsize=(8,4), layout='tight')
plt.xlabel('Hour')
plt.ylabel('Demand [GW]')
ax = plt.gca()
colored_line(demand_TDR.Time_Index, demand_TDR.Demand_MW/1000, colormaps['rainbow'](demand_TDR.Rep_Period/52), ax)
h, l = ax.get_legend_handles_labels()
h.append(MulticolorPatch(colormaps['rainbow'](np.linspace(0,1,11))))
l.append('Representative Weeks')
ax.set_xlim([1,8760])
ax.set_ylim([0,25])
plt.legend(h,l, loc='best', handler_map={MulticolorPatch: MulticolorPatchHandler()})
fig.savefig('plots/demand_TDR.pdf')



results_8736       = f'{case_TDR_demand}/results'
results_TDR_SDS    = f'{case_algorithms}/results_TDR_SDS'
results_TDR_GX_LDS = f'{case_algorithms}/results_TDR_GX_LDS'
results_TDR_SC_LDS = f'{case_algorithms}/results_TDR_SC_LDS'

class model_parameters:
    def __init__(self, results_path):
        with open(f'{results_path}/model_parameters.txt','r') as f:
            lines = f.readlines()
            print('---------------------')
            print(results_path)
            for line in lines: print(line.replace('\n',''))
            print('---------------------\n')
            lines = [line.replace(';','').replace(')','').split() for line in lines]

        self.presolve_rows = np.array(lines[4][4].split('(-'), dtype=int).sum()
        self.presolve_cols = np.array(lines[4][6].split('(-'), dtype=int).sum()
        self.presolve_nzrs = np.array(lines[4][8].split('(-'), dtype=int).sum()
        self.rows = int(lines[6][3])
        self.cols = int(lines[6][5])
        self.nzrs = int(lines[6][8])
        self.variables = int(lines[8][-1])
        self.free = int(lines[9][-1])
        self.equality = int(lines[11][-1])
        self.constraints = int(lines[10][-1])

        status = pd.read_csv(f'{results_path}/status.csv')
        self.status = status.Status
        self.solve_time = status.Solve[0]
        self.objective = status.Objval[0]
        

mp_gx = model_parameters(results_TDR_GX_LDS)
mp_sc   = model_parameters(results_TDR_SC_LDS)
print()
print(f'Parameter count GX - SC')
print(f'Before Presolve')
print(f'Rows: {mp_gx.presolve_rows - mp_sc.presolve_rows}, Cols: {mp_gx.presolve_cols - mp_sc.presolve_cols}, Nonzeros: {mp_gx.presolve_nzrs - mp_sc.presolve_nzrs}')
print(f'IPX')
print(f'Rows: {mp_gx.rows - mp_sc.rows}, Cols: {mp_gx.cols - mp_sc.cols}, Nonzeros: {mp_gx.nzrs - mp_sc.nzrs}')
print(f'Constraints: {mp_gx.constraints - mp_sc.constraints}, Variables: {mp_gx.variables - mp_sc.variables}')
print(f'Free Varaibles: {mp_gx.free - mp_sc.free}, Equality Constraints: {mp_gx.equality - mp_sc.equality}')
print()
print(f'Solve Time: {mp_gx.solve_time} - {mp_sc.solve_time} = {mp_gx.solve_time-mp_sc.solve_time}s')
            
batteries = ['MA_battery', 'CT_battery', 'ME_battery']

# get true optimal SoC given 8736 demand as reconsctucted from the TDR period map
SoC_8736 = pd.read_csv(f'{results_8736}/storage.csv')[2:] # timeseries starts at index 2
SoC_8736['Hour'] = np.arange(8736)+1

# reconstruct 8736 storage SoC from represenative period data
storage_TDR_SDS = pd.read_csv(f'{results_TDR_SDS}/storage.csv')[2:] # timeseries starts at index 2

SoC_TDR_SDS = pd.concat([storage_TDR_SDS[168*(i-1):168*i] for i in period_map.Rep_Period_Index])
SoC_TDR_SDS['Hour'] = np.arange(8736)+1

# get SoC for LDS representations
SoC_TDR_GX_LDS = pd.read_csv(f'{results_TDR_GX_LDS}/storageEvol.csv')[batteries][1:]
SoC_TDR_GX_LDS['Total'] = SoC_TDR_GX_LDS.sum(axis=1)
SoC_TDR_GX_LDS['Hour'] = np.arange(8736)+1

SoC_TDR_SC_LDS = pd.read_csv(f'{results_TDR_SC_LDS}/storageEvol.csv')[batteries][1:]
SoC_TDR_SC_LDS['Total'] = SoC_TDR_SC_LDS.sum(axis=1)
SoC_TDR_SC_LDS['Hour'] = np.arange(8736)+1

# Plot SoC for the different dispatch algorithms
fig = plt.figure(figsize=(8,4), layout='tight')
plt.xlabel('Hour')
plt.ylabel('SoC [MWh]')
plt.plot(SoC_8736.Hour, SoC_8736.Total, label='8736')
plt.plot(SoC_TDR_GX_LDS.Hour, SoC_TDR_GX_LDS.Total, label='11 Rep. Weeks: GenX LDS')
plt.plot(SoC_TDR_SC_LDS.Hour, SoC_TDR_SC_LDS.Total, label='11 Rep. Weeks: Sparse Chronology')
plt.plot(SoC_TDR_SDS.Hour, SoC_TDR_SDS.Total, label='11 Rep. Weeks: SDS')
ax = plt.gca()
ax.set_xlim([1,8736])
plt.legend(loc='best')
fig.savefig('plots/SoC_total_full_year.pdf')

fig = plt.figure(figsize=(8,4), layout='tight')
plt.xlabel('Hour')
plt.ylabel('SoC [MWh]')
plt.plot(SoC_8736.Hour, SoC_8736.Total, label='8736')
plt.plot(SoC_TDR_GX_LDS.Hour, SoC_TDR_GX_LDS.Total, label='11 Rep. Weeks: GenX LDS')
plt.plot(SoC_TDR_SC_LDS.Hour, SoC_TDR_SC_LDS.Total, label='11 Rep. Weeks: Sparse Chronology')
ax = plt.gca()
ax.set_xlim([1,8736])
plt.legend(loc='best')
fig.savefig('plots/SoC_total_full_year_LDS.pdf')

