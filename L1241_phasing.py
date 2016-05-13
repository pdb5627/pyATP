#! /usr/bin/python
# -- coding: utf-8 --

from __future__ import print_function, unicode_literals

import pyATP
import lineZ

import sys, shutil, glob, itertools, codecs, os
import numpy as np
np.set_printoptions(linewidth=120)

# =============================================================================
# Configure file names and bus names

proj_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/'
atp_filenames = ['L1241Phasing2.atp', 'L1241Phasing3.atp']
model_weights = [1., 2.] # Weight no wind more than full wind.

# Run all data in a temporary directory since the ATPDraw won't know
# to rebuild all the .pch files that this program monkeys with.
tmp_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/tmp/'

# =============================================================================
# Enter line configuration information.

# Buses for monitoring voltage unbalance
buses = ['NPLAT', 'THEDF', 'STAPL', 'MAXWE', 'CALLA', 'LOUPC', 'NLOUP', 'BBWIN',
         'CROOK', 'ORD',   'SPALD', 'BROKE', 'MUDDY']
Pos = {}
Str_names = {}

# H-frame structures
Pos['H'] = ['EHS side', 'Center', 'OPGW side']
Str_names['H'] = 'H-frame'

# Single-pole structures
Pos['SP'] = ['Bot', 'Mid', 'Top']
Str_names['SP'] = 'Single-pole'

# Segment lengths & Configurations
sections = [(0,       'SP', 'Muddy Creek Sub',   'L241A'),
            ( 5.98,   'H',  'Transition',        'L241B'),
            ( 6.73,   'SP', 'Transition',        'L241C'),
            (11.69,   'H',  'Transition',        'L241D'),
            (16.16,   'SP*', 'Transition',       'L241E'),
            (23.27,   'H', 'Special Transition', 'L241F'),
            (30.01,   'SP', 'Transition',        'L241G'),
            (39.63,   '',   'Ord Sub', '')]

# =============================================================================
# Define transition possibilities for this study
def transitions(from_str, to_str, bot_to_center = True):
    ''' Returns a list of the possible transitions from from_str to to_str
        bot_to_center indicates whether the transition from delta to horizontal should allow
        the bottom conductor in the delta to move to the center position in the horizontal configuration.
        limit_SP indicates whether the transition from delta to vertical and back to delta should allow
        all phase change combinations or whether only the top two phases should be allowed to swap. Set
        to True to limit the transition to only swapping the top two phases.
        
        _ indicates a non-transposing continuation
        * indicates a transition that sets bot_to_center to true.
    '''
    if from_str[-1] == '_':
        # Non-transposing continuation
        return [(0, 1, 2)]
    if from_str[-1] == '*':
        bot_to_center = True
    from_str2 = from_str.rstrip('_*')
    to_str2 = to_str.rstrip('_*')
    if from_str2 == 'SP' and to_str2 == 'SP':
        # For this study, vertical structures are kept the same top-to-bottom as delta
        # If transition is allowed, it is just inverting positions of the top two
        # faces.
        return ((0, 1, 2), (0, 2, 1))
    if from_str2 == 'H' and to_str2 =='SP':
        return ((1, 0, 2), (2, 0, 1)) if bot_to_center else ((2, 0, 1),)

    if to_str2 == 'H' and from_str2 == 'SP':
        return ((1, 0, 2), (1, 2, 0)) if bot_to_center else ((1, 2, 0),)
    
    if from_str2 =='H' and to_str2 == 'H':
        # H-frame transposition structure lets either outside phase move to center
        return ((0,2,1), (1,0,2), (0,1,2))
    raise ValueError('from_str = '+from_str+', to_str = '+to_str)

# =============================================================================
# Divide data into some different arrays and print output to the screen
    
PIs = np.array([s[0] for s in sections])
PI_desc = [s[2] for s in sections]
section_ATPname = [s[3] for s in sections[:-1]]
# L is an array of the lengths of the segments
md_list = ['### Line Sections:']
for n, s_start, s_end in zip(range(1,len(sections)), sections[:-1], sections[1:]):
    md_list.append('Mile %.3f: %s' % (s_start[0], s_start[2]))
    md_list.append('    Section %d: %s (%.3f mi)' % (n, Str_names[s_start[1].rstrip('_*')], s_end[0] - s_start[0]))
md_list.append('Mile %.3f: %s' % (s_end[0], s_end[2]))
print('\n'.join(md_list))
L = PIs[1:] - PIs[:-1]

str_types = [s[1] for s in sections[:-1]]

transitions_list = [((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0),
                    (2, 0, 1))] + [transitions(from_str, to_str,
                                               bot_to_center=False)
                    for from_str, to_str in zip(str_types[:-1], str_types[1:])]

# Shorter list for debugging program
#transitions_list = [((0, 1, 2), (0, 2, 1))] + [transitions(from_str, to_str,
#                                               bot_to_center=False)
#                    for from_str, to_str in zip(str_types[:-1], str_types[1:])]

# =============================================================================
# Generate list of all possible transitions
all_transitions_dict = lineZ.make_transitions_dict(transitions_list, str_types)
print('Number of phasing combinations by number of transpositions:')

all_transitions_list = []
for k, v in all_transitions_dict.items():
    print(k, len(v))
    all_transitions_list.extend(v)
                     
# =============================================================================
# Define evaluation criteria and filter to get the non-dominated results.        
def max_neg_seq_unbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance) '''
    rtn = np.max(r[2])
    if not as_str:
        return rtn
    else:
        return 'Maximum neg. seq. voltage unbalance: %.4f %%' % rtn
max_neg_seq_unbalance.description = 'Max. Negative Seq. Unbalance'
max_neg_seq_unbalance.units = '%'
    
def avg_neg_seq_unbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance) '''
    rtn = np.mean(r[2])
    if not as_str:
        return rtn
    else:
        return 'Average neg. seq. voltage unbalance: %.4f %%' % rtn
avg_neg_seq_unbalance.description = 'Avg. Negative Seq. Unbalance'
avg_neg_seq_unbalance.units = '%'

#criteria = [max_neg_seq_unbalance, avg_neg_seq_unbalance]
criteria = [max_neg_seq_unbalance, avg_neg_seq_unbalance]
criteria_weights = [1., 1.] # Weight avg. more to scale it up.

# =============================================================================
# Without L1241 in model.
# ATP LIS file is saved from ATPDraw run and should not be over-written.

r_noL1241 = pyATP.process_SS_bus_voltages(proj_dir + 'L1241Phasing_noL1241.lis',
                                          buses, RMS_scale = True)
ph_voltages, seq_voltages, neg_seq_unbalance = r_noL1241
print('-'*80)
print('Without L1241 in the model:')
print('Phase voltages and negative-sequence unbalance voltage')
for n, b in enumerate(buses):
    print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
    
for c in criteria:
    print(c(r_noL1241, as_str=True))
    
# Hold results of each model in a dict indexed by the ATP model name
results_dict = {l: {} for l in all_transitions_list}
r_base = {'base': {}}

for atp_filename in atp_filenames:
    # =============================================================================
    # Copy working files to temp directory
    
    # Main ATP model
    shutil.copyfile(proj_dir+atp_filename, tmp_dir+atp_filename)
    
    # Line segment data files
    for f in itertools.chain(glob.glob(proj_dir + '*.dat'),
                             glob.glob(proj_dir + '*.lib')):
        shutil.copy(f, tmp_dir)
    
    ATP_file = tmp_dir + atp_filename
    LIS_file = pyATP.lis_filename(ATP_file)
    
    # Find/replace directory name in ATP file so .lib files will be included properly.
    with codecs.open(ATP_file,'r', encoding='cp1252', errors='replace') as f:
        filedata = f.read()
    
    find = os.path.abspath(proj_dir)
    replace = os.path.abspath(tmp_dir)
    
    newdata = filedata.replace(find, replace)
    
    with codecs.open(ATP_file,'w', encoding='cp1252', errors='replace') as f:
        f.write(newdata)
        
    # =============================================================================
    # Run the model before any changes are made by the program as a sanity check.
    
    print('-'*80)
    print('With L1241 in the model, phased as in the model:')
    print('Phase voltages and negative-sequence unbalance voltage')
    pyATP.run_ATP(ATP_file)
    
    ph_voltages, seq_voltages, neg_seq_unbalance = pyATP.process_SS_bus_voltages(LIS_file, buses, RMS_scale = True)
    r_base['base'][atp_filename] = ((ph_voltages, seq_voltages, neg_seq_unbalance),)
    
    for n, b in enumerate(buses):
        print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
        
    for c in criteria:
        print(c(r_base['base'][atp_filename][0], as_str=True))
    
    print('-'*80)
    
    # =============================================================================
    # Run analysis of all data cases. 
    # Time to run is approximately 1-2 s per case for this model.    
    results = []
    print('Running calcs....')
    for n, t in all_transitions_dict.items():
        for n2, l in enumerate(t):
            print('For %d transpositions, case %d of %d' % (n, n2, len(t)))
            
            # Set phasing of line sections in ATP .dat files & re-run line constants.
            for Pt, s in zip(lineZ.cum_Pt(l), section_ATPname):
                line_const = tmp_dir + s + '.dat'
                
                with open(line_const, 'r') as f:
                    inlines = f.readlines()
                line_data = pyATP.LineConstCards()
                line_data.read(inlines)
                
                # Modify the phasing in .dat file.
                for idx, ph in enumerate(Pt):
                    line_data.data['conductors'][idx]['IP'] = ph + 1
                
                outlines = line_data.write()
                
                with open(line_const, 'w') as f:
                    f.writelines(outlines)
                # Run ATP on the .dat file to create .pch file.
                pyATP.run_ATP(line_const)
                
                # Make .lib file
                line_pch = tmp_dir + s + '.pch'
                with open(line_pch, 'r') as f:
                    pchlines = f.readlines()
                liblines = pyATP.make_ATPline_lib(pchlines)
                line_lib = tmp_dir + s + '.lib'
                with open(line_lib, 'w') as f:
                    f.writelines(liblines)
                
            # Run main ATP model with modified line sections
            pyATP.run_ATP(ATP_file)
            ph_voltages, seq_voltages, neg_seq_unbalance = pyATP.process_SS_bus_voltages(LIS_file, buses)
            
            # Save results to array
            #results.append((n, l, (ph_voltages, seq_voltages, neg_seq_unbalance)))
    
    
            results_dict[l][atp_filename] = ((ph_voltages, seq_voltages, neg_seq_unbalance),)
    
# =============================================================================
# Filter to non-dominated results across all models.

filtered_results_dict = lineZ.filter_nondominated_results_multimodel(results_dict, criteria)

# Compute weighted criteria results
soln_list, weighted_results = lineZ.apply_criteria_weighting(results_dict, criteria, model_weights, criteria_weights)
weights_results_dict = {soln: wt for soln, wt in zip(soln_list, weighted_results)}
  
# =============================================================================
# Print results

# Base case
print('-'*80)
print('With L1241 in the model, phased as in the model:')
print('Phase voltages and negative-sequence unbalance voltage')
for model in atp_filenames:
    print('Model: %s' % atp_filenames)
    ph_voltages, seq_voltages, neg_seq_unbalance = r_base['base'][atp_filename][0]

    for n, b in enumerate(buses):
        print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
    
    for c in criteria:
        print(c(r_base['base'][atp_filename][0], as_str=True))
_, base_weighted_results = lineZ.apply_criteria_weighting(r_base, criteria, model_weights, criteria_weights)
print('Weighted results: %.4f' % base_weighted_results)

print('-'*80)
print('Results of best option(s)') 
best_soln = {'best': None} 
#for soln, r in filtered_results_dict.items():
for soln, wt in sorted(zip(soln_list, weighted_results), key=lambda k: k[1]):
    r = results_dict[soln]
    if best_soln['best'] is None:
        best_soln['best'] = r
    print('-'*80)
    #ph_voltages, seq_voltages, neg_seq_unbalance = r[0]
    print(soln[0], soln[5])
    for model in atp_filenames:
        for c in criteria:
            print('%s, %s'% (model, c(r[model][0], as_str=True)))
    print('Weighted results: %.4f' % wt)
    #continue
    
    #for n, b in enumerate(buses):
    #    print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
    
    phasing_info = lineZ.Pt_list_to_phasing(soln, str_types, Pos, Phase_list=('A', 'B', 'C'))
    print('Line Sections:')
    for n, phasing, s_start, s_end in zip(range(1,len(sections)), phasing_info, sections[:-1], sections[1:]):
        print(('Mile %.3f: %s' % (s_start[0], s_start[2])) + (' (Transposition)' if n > 1 and sections[n-2][1].rstrip('_*') == sections[n-1][1].rstrip('_*') and r[1][n-1]!=(0, 1, 2) else ''))
        print('Section %d: %s (%.3f mi)' % (n, Str_names[s_start[1].rstrip('_*')], s_end[0] - s_start[0]))
        print(phasing)
    print('Mile %.3f: %s' % (s_end[0], s_end[2]))

print('-'*80)
    
# Print dominated solutions. 
dominated_solns = [soln for soln in all_transitions_list if soln not in filtered_results_dict ]
print('Dominated solutions:')
for soln in dominated_solns:
    print(soln[0], soln[5])
    
# =============================================================================
# Utility function and variables for plotting
def apply_function_to_results(results_dict, model, f):
    return np.array([f(results_dict[soln][model][0]) for soln in results_dict.keys()])

soln_list = list(results_dict.keys())
# Save list of models to ensure we iterate over them in a consistent order.
model_list = atp_filenames

# Makes a list of tuples of all combinations.
plot_list = list(itertools.product(model_list, criteria))
plot_desc = ['%s\n%s (%s)' % (fx.description, mx, fx.units) for mx, fx in plot_list]
# =============================================================================
# Create plot if matplotlib is available

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit()


ncriteria = len(model_list)*len(criteria)
nplot = 1
plt.figure()
plt.gcf().subplotpars.update(hspace=0.05, wspace=0.05)
plt.axis('tight')
for my in model_list:
    for fy in criteria:
        for mx in model_list:
            for fx in criteria:
                plt.subplot(ncriteria, ncriteria, nplot)
    

                
                xvals = apply_function_to_results(best_soln, mx, fx)
                yvals = apply_function_to_results(best_soln, my, fy)
                #print(xvals, yvals)
                l3 = plt.scatter(xvals, yvals, c='orange', s=150., label='Weighted Best')
                
                Pt_brad = ((1, 2, 0), (1, 2, 0), (2, 0, 1), (1, 2, 0), (2, 0, 1), (1, 0, 2), (2, 0, 1))
                Brad_results = {Pt_brad: results_dict[Pt_brad]}
                xvals = apply_function_to_results(Brad_results, mx, fx)
                yvals = apply_function_to_results(Brad_results, my, fy)
                l4 = plt.scatter(xvals, yvals, c='g', s=150., label='Brad\'s Proposed Phasing')

                xvals = np.array(fx(r_base['base'][mx][0]))
                yvals = np.array(fy(r_base['base'][my][0]))
                l5 = plt.scatter(xvals, yvals, c='m', s=80., label='In base model')
                
                xvals = apply_function_to_results(results_dict, mx, fx)
                yvals = apply_function_to_results(results_dict, my, fy)
                l1 = plt.scatter(xvals, yvals, c='b', s=1., label='All results')
                #texts = []
                #for x, y, label in zip(xvals, yvals, results_dict.keys()):
                #    texts.append(plt.text(x, y, '%s\n%s' % (label[0], label[5]), size=8))
                #    plt.gca().annotate('%s\n%s' % (label[0], label[5]), xy=(x, y), textcoords='data')
                
                xrange = np.max(xvals) - np.min(xvals)
                yrange = np.max(yvals) - np.min(yvals)
                m = 0.05
                plt.xlim([np.min(xvals) - xrange*m, np.max(xvals) + xrange*m])
                plt.ylim([np.min(yvals) - yrange*m, np.max(yvals) + yrange*m])


                xvals = apply_function_to_results(filtered_results_dict, mx, fx)
                yvals = apply_function_to_results(filtered_results_dict, my, fy)
                # Idea: color by ranking rather than value??                
                cvals = np.log(np.log(np.array([weights_results_dict[soln] for soln in filtered_results_dict.keys()])))
                l2 = plt.scatter(xvals, yvals, c=cvals, cmap='jet',
                                 s=120., label='Non-dominated solutions')

                plt.tick_params(axis='both', which='major', labelsize='small')

                # Show y-axis tic labels and axis label only on first column
                if mx == model_list[0] and fx ==criteria[0]:
                    plt.ylabel('%s\n%s (%s)' % (fy.description, my, fy.units), fontsize='small')
                else:
                    plt.gca().set_yticklabels([])
                # Show x-axis tic labels and axis label only on last row
                if my == model_list[-1] and fy ==criteria[-1]:
                    plt.xlabel('%s\n%s (%s)' % (fx.description, mx, fx.units), fontsize='small')
                else:
                    plt.gca().set_xticklabels([])
                
                nplot += 1

plt.gcf().legend((l1, l2, l3, l4, l5), ('Dominated', 'Non-dominated', 'Weighted Best', 'Brad\'s proposal', 'In base model'),
                 scatterpoints=1, fontsize='small',
                 #loc=(1.1, 0.5))#, 
                 bbox_to_anchor=(1, 0.5))
plt.colorbar()

plt.show()
'''

# =============================================================================
# Create plot if bokeh is available
try:
    from bokeh.document import Document
    from bokeh.embed import file_html
    from bokeh.models.glyphs import Circle
    from bokeh.models import (
        BasicTicker, ColumnDataSource, Grid, GridPlot, LinearAxis,
        DataRange1d, PanTool, Plot, WheelZoomTool
    )
    from bokeh.resources import INLINE
    from bokeh.sampledata.iris import flowers
    from bokeh.util.browser import view
except ImportError:
    sys.exit()
    


colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

flowers['color'] = flowers['species'].map(lambda x: colormap[x])


source = ColumnDataSource(
    data = { k: apply_function_to_results(results_dict, k[0], k[1]) for k in plot_list }
)
source.add(data=results_dict.keys(), name='soln')

xdr = DataRange1d(bounds=None)
ydr = DataRange1d(bounds=None)

def make_plot(xname, yname, xax=False, yax=False):
    mbl = 40 if yax else 0
    mbb = 40 if xax else 0
    plot = Plot(
        x_range=xdr, y_range=ydr, background_fill_color="#efe8e2",
        border_fill_color='white', title="", h_symmetry=False, v_symmetry=False,
        plot_width=200 + mbl, plot_height=200 + mbb, min_border_left=2+mbl, min_border_right=2,
        min_border_top=2, min_border_bottom=2+mbb)

    circle = Circle(x=xname, y=yname, fill_color="color", fill_alpha=0.2, size=4, line_color="color")
    r = plot.add_glyph(source, circle)

    xdr.renderers.append(r)
    ydr.renderers.append(r)

    xticker = BasicTicker()
    if xax:
        xaxis = LinearAxis()
        plot.add_layout(xaxis, 'below')
        xticker = xaxis.ticker
    plot.add_layout(Grid(dimension=0, ticker=xticker))

    yticker = BasicTicker()
    if yax:
        yaxis = LinearAxis()
        plot.add_layout(yaxis, 'left')
        yticker = yaxis.ticker
    plot.add_layout(Grid(dimension=1, ticker=yticker))

    plot.add_tools(PanTool(), WheelZoomTool())

    return plot

xattrs = plot_desc
yattrs = xattrs
plots = []
xranges = [None]*len(xattrs)
yranges = [None]*len(yattrs)

for y in yattrs:
    row = []
    for x in xattrs:
        xax = (y == yattrs[-1])
        yax = (x == xattrs[0])
        plot = make_plot(x, y, xax, yax)
        row.append(plot)
    plots.append(row)

grid = GridPlot(children=plots)

doc = Document()
doc.add_root(grid)

if __name__ == "__main__":
    filename = "output/L1241_phasing_results.html"
    with open(filename, "w") as f:
        f.write(file_html(doc, INLINE, "Scatterplot Matrix"))
    print("Wrote %s" % filename)
    view(filename)
'''