from matplotlib import pyplot as pp
import numpy as np
import DNA_manipulations as DNAm

#Places labels on top of bars within bar graph
def autolabel(bins, plotType, integer = True):
    # attach some text labels
    offSet = max([b.get_height() for b in bins])
    for bins in bins:
        height = bins.get_height()
        if integer:
            plotType.text(bins.get_x()+bins.get_width()/2., -(height + offSet*0.025), '-%d'%int(height),
                    ha='center', va='bottom')
        else:
            plotType.text(bins.get_x()+bins.get_width()/2., -(height + offSet*0.025), '-%.2f'%(height),
                    ha='center', va='bottom')

# inporting training data
DNA1, recSite1, freq1 = DNAm.array("sixtyninemers_frequencies_GsAK.csv")
DNA2, recSite2, freq2 = DNAm.array("sixtyninemers_frequencies_TnAK.csv")
DNA3, recSite3, freq3 = DNAm.array("sixtyninemers_frequencies_BgAK.csv")
DNA4, recSite4, freq4 = DNAm.array("sixtyninemers_frequencies_BsAK.csv")

propeller1 = [DNAm.propeller(seq, overhang = 12)[1] for seq in DNA1[0:600]]
propeller2 = [DNAm.propeller(seq, overhang = 12)[1] for seq in DNA2[0:600]]
propeller3 = [DNAm.propeller(seq, overhang = 12)[1] for seq in DNA3[0:600]]
propeller4 = [DNAm.propeller(seq, overhang = 12)[1] for seq in DNA4[0:600]]

freq = freq1[0:600] + freq2[0:600] + freq3[0:600] + freq4[0:600]
propeller = propeller1 + propeller2 + propeller3 + propeller4
propellerAverageAll = np.mean(propeller, axis = 0)
propellerGeneAverages = [np.mean(propeller1), np.mean(propeller2), np.mean(propeller3), np.mean(propeller4)]

orderedpropeller = sorted(zip(freq, propeller), key = lambda x: int(x[0]))
orderedpropeller = [x[1] for x in orderedpropeller]
# orderedpropellerFreq = [x[0] for x in orderedpropeller]

# fig = pp.figure()
# ax = pp.subplot()
# ax.plot(orderedpropellerFreq)
# pp.show()

l = len(orderedpropeller)

averagePropellerTopTen = np.mean(orderedpropeller[l - int(l*0.15):l], axis = 0)
averagePropellerBottomTen = np.mean(orderedpropeller[0:int(l*0.15)], axis = 0)

bpRange = range(-12,0)
reversebpRange = range(-12,0)
reversebpRange.reverse()
xlabels = bpRange + ['N1', 'N2', 'N3', 'N4', 'N5'] + reversebpRange

fig = pp.figure(figsize = (12, 7))
ax = pp.subplot2grid((2,4), (0,0), colspan = 4)
ax.plot(propellerAverageAll, label = 'Average propeller twist', linewidth = 2, color = 'gray')
ax.plot(averagePropellerTopTen, label = 'Top 10% insert freq.', marker = '*')
ax.plot(averagePropellerBottomTen, label = 'Bottom 10% insert freq.', marker = '*')
ax.set_ylim([0,-17])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Average positional propeller twist")
ax.set_xlabel("Nucleotide position")
ax.set_ylabel(r"Average propeller twist ($\degree$)")
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)

ax = pp.subplot2grid((2,4), (1,0))
n_bar = len(propellerGeneAverages)
ind = np.arange(n_bar)

p1 = ax.bar(ind,propellerGeneAverages, width = 0.8)
ax.set_ylim([0,-17])
ax.set_xticks(range(len(propellerGeneAverages)))
ax.set_xticklabels(["GsAK", "TnAK", "BgAK", "BsAK"])
ax.margins(0.05)
ax.set_title("Average gene propeller twist")
ax.set_xlabel("Gene")
ax.set_ylabel(r"Average propeller twist ($\degree$)")
autolabel(p1, ax, integer = False)


ax = pp.subplot2grid((2,4), (1,1), colspan = 3)
ax.plot(propeller[92], label = 'High insert freq.', marker = '*')
ax.plot(propeller[0], label = 'Low insert freq.', marker = '*')
ax.set_ylim([0,-17])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Representative low and high freq. propeller twist")
ax.set_xlabel("Nucleotide position")

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False, loc = 'upper right')


pp.tight_layout()
fig.savefig('Propeller Analysis_v2.pdf')


