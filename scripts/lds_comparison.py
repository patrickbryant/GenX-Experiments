import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
import yaml

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

# index_power = ['battery_discharge',    'battery_charge', 'natural_gas_combined_cycle', 'solar_pv', 'onshore_wind']
# names_power = ['Battery Discharge',    'Battery Charge',                'Natural Gas',   'Solar',          'Wind']#, 'Curtailed Wind', 'Curtailed Solar', 'Generation',  'Demand', 'Transmission12', 'Transmission13']
# color_power = [          '#BB2777',           '#BB568D',                    '#4C2469', '#F7CB46',       '#52B3EA']#,        '#AFD5EA',         '#F7E8B9',    '#FF0000', '#0000FF',        '#666666',        '#333333']

index_power = ['battery_charge',    'battery_discharge', 'natural_gas_combined_cycle', 'solar_pv', 'onshore_wind', 'curtail_onshore_wind', 'curtail_solar_pv']
names_power = ['Battery Charge',    'Battery Discharge',                'Natural Gas',   'Solar',          'Wind',       'Curtailed Wind',  'Curtailed Solar']#, 'Generation',  'Demand', 'Transmission12', 'Transmission13']
color_power = [          '#BB568D',           '#BB2777',                    '#4C2469', '#F7CB46',       '#52B3EA',              '#AFD5EA',          '#F7E8B9']#,    '#FF0000', '#0000FF',        '#666666',        '#333333']
index_curtail = [  'onshore_wind',        'solar_pv']
names_curtail = ['Curtailed Wind', 'Curtailed Solar']
color_curtail = [       '#AFD5EA',         '#F7E8B9']

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
colored_line(demand_raw.Time_Index, demand_raw.Demand_MW/1000, colormaps['hsv'](demand_raw.Week/52)*np.array(demand_raw.In_Rep_Period).reshape(-1,1), ax)
h, l = ax.get_legend_handles_labels()
h.append(MulticolorPatch(colormaps['hsv'](np.linspace(0,1,11))))
l.append('Representative Weeks')
ax.set_xlim([1,8760])
ax.set_ylim([0,25])
plt.legend(h,l, loc='best', handler_map={MulticolorPatch: MulticolorPatchHandler()})
fig.savefig('plots/demand_raw.pdf')

fig = plt.figure(figsize=(8,4), layout='tight')
plt.xlabel('Hour')
plt.ylabel('Demand [GW]')
ax = plt.gca()
colored_line(demand_TDR.Time_Index, demand_TDR.Demand_MW/1000, colormaps['hsv'](demand_TDR.Rep_Period/52), ax)
h, l = ax.get_legend_handles_labels()
h.append(MulticolorPatch(colormaps['hsv'](np.linspace(0,1,11))))
l.append('Representative Weeks')
ax.set_xlim([1,8760])
ax.set_ylim([0,25])
plt.legend(h,l, loc='best', handler_map={MulticolorPatch: MulticolorPatchHandler()})
fig.savefig('plots/demand_TDR.pdf')


class results:
    def __init__(self, results_path):
        self.results_path = results_path
        self.case_path = '/'.join(self.results_path.split('/')[:-1])
        with open(f'{self.results_path}/run_settings.yml', 'r') as file:
            self.run_settings = yaml.safe_load(file)

        self.power_csv   = pd.read_csv(f'{results_path}/power.csv' )     [2:].reset_index(drop=True)
        self.charge_csv  = pd.read_csv(f'{results_path}/charge.csv')     [2:].reset_index(drop=True)
        self.curtail_csv = pd.read_csv(f'{results_path}/curtailment.csv')[2:].reset_index(drop=True)
        self.flow_csv    = pd.read_csv(f'{results_path}/flow.csv')
        self.flow_csv[['1','2']] /= 1000 # scale to GW
        self.balance_csv = pd.read_csv(f'{results_path}/power_balance.csv')[2:].reset_index(drop=True)
        self.price_csv   = pd.read_csv(f'{results_path}/prices.csv') # $/MWh

        self.capacity_csv = pd.read_csv(f'{results_path}/capacity.csv')
        total_battery = self.capacity_csv[self.capacity_csv.Resource.str.contains('battery')].sum(axis=0)
        total_battery.Resource = 'battery'
        self.capacity_csv.loc[len(self.capacity_csv)] = total_battery

        try:
            self.storage = pd.read_csv(f'{results_path}/StorageEvol.csv')[1:].reset_index(drop=True)
            self.LDS = True
        except FileNotFoundError:
            self.storage_csv = pd.read_csv(f'{results_path}/storage.csv')[2:].reset_index(drop=True)
            self.storage_csv.drop('Total', axis=1, inplace=True)
            self.LDS = False

        self.network = pd.read_csv(f'{self.case_path}/system/Network.csv')
        self.zones = self.network[self.network.columns[0]]
        self.resources = list(set(['_'.join(col.split('_')[1:]) for col in self.power_csv.columns if col[2]=='_']))
        for r in self.resources:
            for z in self.zones:
                if f'{z}_{r}' not in self.power_csv.columns:
                    self.power_csv  [f'{z}_{r}'] = np.zeros(self.power_csv  .shape[0])
                    self.curtail_csv[f'{z}_{r}'] = np.zeros(self.curtail_csv.shape[0])
                self.power_csv  [f'{z}_{r}'] /= 1000 # convert to GW
                self.curtail_csv[f'{z}_{r}'] /= 1000 # convert to GW

            self.power_csv  [r] = self.power_csv  [['_'.join([z,r]) for z in self.zones]].sum(axis=1)
            self.curtail_csv[r] = self.curtail_csv[['_'.join([z,r]) for z in self.zones]].sum(axis=1)
            
        self.power_csv['battery_discharge'] =  self.power_csv.battery
        self.power_csv['battery_charge']    = -self.charge_csv.Total/1000 # convert to GW
        for i, z in enumerate(self.zones):
            self.power_csv[f'{z}_battery_charge']    = -self.charge_csv[f'{z}_battery']/1000 # convert to GW
            self.power_csv[f'{z}_battery_discharge'] =  self. power_csv[f'{z}_battery']
            self.power_csv[f'{z}_demand']            = -self.balance_csv[f'Demand{"."+str(i) if i else ""}']/1000
            self.power_csv[f'{z}_exports']           =  self.balance_csv[f'Transmission_NetExport{"."+str(i) if i else ""}']/1000
            self.power_csv[f'{z}_imports']           = self.power_csv[f'{z}_exports']
            self.power_csv.loc[self.power_csv[f'{z}_exports']>0,f'{z}_exports']=0
            self.power_csv.loc[self.power_csv[f'{z}_imports']<0,f'{z}_imports']=0

        self.power_csv['transmission12'] = self.flow_csv['1']
        self.power_csv['transmission13'] = self.flow_csv['2']        

        # reconstruct full time series from TDR period map
        if self.run_settings['TimeDomainReduction']:
            self.TDR_path = self.case_path+'/'+self.run_settings["TimeDomainReductionFolder"]
            with open(f'{self.TDR_path}/time_domain_reduction_settings.yml', 'r') as file:
                self.time_domain_reduction_settings = yaml.safe_load(file)
            self.N_trp = self.time_domain_reduction_settings['TimestepsPerRepPeriod']
            self.period_map = pd.read_csv(f'{self.TDR_path}/Period_map.csv')
            self.power   = pd.concat([self.power_csv  [self.N_trp*(i-1):self.N_trp*i] for i in self.period_map.Rep_Period_Index]).reset_index(drop=True)
            self.curtail = pd.concat([self.curtail_csv[self.N_trp*(i-1):self.N_trp*i] for i in self.period_map.Rep_Period_Index]).reset_index(drop=True)
            self.price   = pd.concat([self.price_csv  [self.N_trp*(i-1):self.N_trp*i] for i in self.period_map.Rep_Period_Index]).reset_index(drop=True)

            if not self.LDS:
                self.storage = pd.concat([self.storage_csv[self.N_trp*(i-1):self.N_trp*i] for i in self.period_map.Rep_Period_Index]).reset_index(drop=True)
        else:
            self.power = self.power_csv
            self.storage = self.storage_csv
            self.curtail = self.curtail_csv
            self.price = self.price_csv
            
        for r in index_curtail:
            self.power[f'curtail_{r}'] = self.curtail[r]
            for z in self.zones:
                self.power[f'{z}_curtail_{r}'] = self.curtail[f'{z}_{r}']
            
        self.power['hour'] = np.arange(self.power.shape[0])+1
        self.storage.drop('Resource', axis=1, inplace=True)
        self.storage['battery'] = self.storage.sum(axis=1)
        self.storage /= 1000 # convert to GWh
        self.power['demand'] = self.power[[f'{z}_demand' for z in self.zones]].sum(axis=1)

        self.battery_revenue = pd.DataFrame() #= self.price[['1','2','3']].copy()
        for i,z in zip(self.zones.index+1, self.zones):
            self.battery_revenue[f'{z}_battery'] = ( self.price[str(i)]*(self.power[f'{z}_battery_charge'] + self.power[f'{z}_battery_discharge'])*1000 ).cumsum(axis=0)
        self.battery_revenue['battery'] = self.battery_revenue.sum(axis=1)

            
    def model_status(self):
        with open(f'{self.results_path}/model_parameters.txt','r') as f:
            lines = f.readlines()
            print('---------------------')
            print(self.results_path)
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

        status = pd.read_csv(f'{self.results_path}/status.csv')
        self.status = status.Status
        self.solve_time = status.Solve[0]
        self.objective = status.Objval[0]

        
    def plot_dispatch_window(self, tstart, tend, plot_name, zone=''):
        fig = plt.figure(figsize=(8.45,4), layout='tight')
        plt.xlabel('Hour')
        plt.ylabel('Power [GW]')
        resources_neg, resources_pos = index_power[0:1], index_power[1: ]
        labels_neg,    labels_pos    = names_power[0:1], names_power[1: ]
        colors_neg,    colors_pos    = color_power[0:1], color_power[1: ]
        if zone:
            resources_neg, resources_pos = [f'{zone}_battery_charge', f'{zone}_exports'], [f'{zone}_{r}' for r in index_power[1:]]
            resources_pos.insert(3, f'{zone}_imports')
            labels_neg.append('Exports')
            labels_pos.insert(3,'Imports')
            colors_neg.append('#666666')
            colors_pos.insert(3,'#999999')
        plt.stackplot(np.array(self.power.hour.loc[tstart:tend]), np.array(self.power.loc[tstart:tend][resources_neg]).transpose(), labels=labels_neg, colors=colors_neg)
        plt.stackplot(np.array(self.power.hour.loc[tstart:tend]), np.array(self.power.loc[tstart:tend][resources_pos]).transpose(), labels=labels_pos, colors=colors_pos)
        plt.plot(self.power.hour.loc[tstart:tend], self.power.loc[tstart:tend][f'{zone}_demand' if zone else 'demand'],   label='Demand',     color='#0000FF')
        axl = plt.gca()
        # ax.set_ylim([-1000,4500])
        axl.set_xlim([tstart+1,tend])
        axr = axl.twinx()
        axr.set_ylabel('Battery SoC [GWd]')
        battery = f'{zone}_battery' if zone else 'battery'
        axr.plot(self.power.hour.loc[tstart:tend], self.storage.loc[tstart:tend][battery]/24, color=(0,0,0,0.8))
        axr.set_ylim([0,self.capacity_csv[self.capacity_csv.Resource==battery].EndEnergyCap.iloc[0]/24_000])
        axl.plot([],[],color=(0,0,0,0.8), label='Battery SoC')
        
        axl.legend(bbox_to_anchor=(1.10, 1), loc='upper left')
        fig.savefig(plot_name)
        plt.close()


        
results_8736       = f'{case_TDR_demand}/results_added_NG'
results_TDR_SDS    = f'{case_algorithms}/results_TDR_SDS'
# results_TDR_GX_LDS = f'{case_algorithms}/results_TDR_GX_LDS'
results_TDR_SC_LDS = f'{case_algorithms}/results_TDR_SC_LDS_added_NG'
results_TDR_GX_LDS = f'{case_algorithms}/results_TDR_GX_LDS_added_NG'
# results_TDR_SC_LDS = f'{case_algorithms}/results_TDR_SC_LDS_vS_unrestricted'

gx = results(results_TDR_GX_LDS)
sc = results(results_TDR_SC_LDS)
gx.model_status()
sc.model_status()
print()
print(f'Parameter count GX - SC')
print(f'Before Presolve')
print(f'Rows: {gx.presolve_rows - sc.presolve_rows}, Cols: {gx.presolve_cols - sc.presolve_cols}, Nonzeros: {gx.presolve_nzrs - sc.presolve_nzrs}')
print(f'IPX')
print(f'Rows: {gx.rows - sc.rows}, Cols: {gx.cols - sc.cols}, Nonzeros: {gx.nzrs - sc.nzrs}')
print(f'Constraints: {gx.constraints - sc.constraints}, Variables: {gx.variables - sc.variables}')
print(f'Free Varaibles: {gx.free - sc.free}, Equality Constraints: {gx.equality - sc.equality}')
print()
print(f'Solve Time: {gx.solve_time} - {sc.solve_time} = {gx.solve_time-sc.solve_time}s')

# get true optimal SoC given full 8736 hourly demand as reconsctucted from the TDR period map
hr = results(results_8736)

tstart, tend = 0,168
# tstart, tend = 44*168, 45*168
hr.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_hr.pdf')
hr.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_hr_1.pdf', zone='MA')
hr.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_hr_2.pdf', zone='CT')
hr.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_hr_3.pdf', zone='ME')

gx.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_gx.pdf')
gx.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_gx_1.pdf', zone='MA')
gx.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_gx_2.pdf', zone='CT')
gx.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_gx_3.pdf', zone='ME')

sc.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_sc.pdf')
sc.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_sc_1.pdf', zone='MA')
sc.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_sc_2.pdf', zone='CT')
sc.plot_dispatch_window(tstart, tend, 'plots/dispatch_window_sc_3.pdf', zone='ME')


color_hr = '#00A400'
color_gx = '#FF7400'
color_sc = '#0078B8'

# Plot SoC for the different dispatch algorithms
def plot_SoC(name, y, title=''):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,7), layout='tight', gridspec_kw={'height_ratios': [3, 1]})
    ax2.set_xlabel('Hour')
    ax1.set_ylabel('SoC [GWd]')
    ax2.set_ylabel('Hourly - TDR [GWd]')
    ax1.plot(hr.power.hour, hr.storage[y]/24, color=color_hr, label='8736')
    ax1.plot(gx.power.hour, gx.storage[y]/24, color=color_gx, label='TDR: GenX LDS')
    ax1.plot(sc.power.hour, sc.storage[y]/24, color=color_sc, label='TDR: Sparse Chronology')
    energy_cap =  hr.capacity_csv[hr.capacity_csv.Resource == y].EndEnergyCap.iloc[0]/24_000
    ax1.plot([1,8736], [energy_cap, energy_cap], color=[0,0,0,0.5], linestyle='--')
    ax1.plot([1,8736], [0, 0], color=[0,0,0,0.5], linestyle='--')
    ax1.set_xlim([1,8736])
    ax2.plot(hr.power.hour, (hr.storage[y] - gx.storage[y])/24, color=color_gx)
    ax2.plot(hr.power.hour, (hr.storage[y] - sc.storage[y])/24, color=color_sc)
    ax1.legend(loc='best')
    fig.savefig(f'plots/{name}.pdf')

plot_SoC('SoC_total_full_year_LDS', 'battery', 'All Batteries')
plot_SoC('SoC_1_full_year_LDS', 'MA_battery', 'Zone 1 Battery')
plot_SoC('SoC_2_full_year_LDS', 'CT_battery', 'Zone 2 Battery')
plot_SoC('SoC_3_full_year_LDS', 'ME_battery', 'Zone 3 Battery')

# Plot battery revenue for the different dispatch algorithms
def plot_battery_revenue(name, y, title=''):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,7), layout='tight', gridspec_kw={'height_ratios': [3, 1]})
    ax2.set_xlabel('Hour')
    ax1.set_ylabel('Battery Net Revenue')
    ax2.set_ylabel('Hourly - TDR')
    ax1.plot(hr.power.hour, hr.battery_revenue[y]/24, color=color_hr, label='8736')
    ax1.plot(gx.power.hour, gx.battery_revenue[y]/24, color=color_gx, label='TDR: GenX LDS')
    ax1.plot(sc.power.hour, sc.battery_revenue[y]/24, color=color_sc, label='TDR: Sparse Chronology')
    ax1.set_xlim([1,8736])
    ax2.plot(hr.power.hour, (hr.battery_revenue[y] - gx.battery_revenue[y])/24, color=color_gx)
    ax2.plot(hr.power.hour, (hr.battery_revenue[y] - sc.battery_revenue[y])/24, color=color_sc)
    ax1.legend(loc='best')
    fig.savefig(f'plots/{name}.pdf')

plot_battery_revenue('battery_revenue_total_full_year_LDS', 'battery', 'All Batteries')
plot_battery_revenue('battery_revenue_1_full_year_LDS', 'MA_battery', 'Zone 1 Battery')
plot_battery_revenue('battery_revenue_2_full_year_LDS', 'CT_battery', 'Zone 2 Battery')
plot_battery_revenue('battery_revenue_3_full_year_LDS', 'ME_battery', 'Zone 3 Battery')
