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
            plotType.text(bins.get_x()+bins.get_width()/2., height + offSet*0.025, '%d'%int(height),
                    ha='center', va='bottom')
        else:
            plotType.text(bins.get_x()+bins.get_width()/2., height + offSet*0.025, '%.2f'%(height),
                    ha='center', va='bottom')

# inporting training data
DNA1, recSite1, freq1 = DNAm.array("sixtyninemers_frequencies_GsAK.csv")
DNA2, recSite2, freq2 = DNAm.array("sixtyninemers_frequencies_TnAK.csv")
DNA3, recSite3, freq3 = DNAm.array("sixtyninemers_frequencies_BgAK.csv")
DNA4, recSite4, freq4 = DNAm.array("sixtyninemers_frequencies_BsAK.csv")

DNA = DNA1[0:600] + DNA2[0:600] + DNA3[0:600] + DNA4[0:600]
freq = freq1[0:600] + freq2[0:600] + freq3[0:600] + freq4[0:600]

GCContent = [DNAm.GC_content(seq, overhang = 12)[1] for seq in DNA]
roll = [DNAm.roll(seq, overhang = 12)[1] for seq in DNA]
mGroove = [DNAm.minorGroove(seq, overhang = 12)[1] for seq in DNA]
helixTwist = [DNAm.helixTwist(seq, overhang = 12)[1] for seq in DNA]
propeller = [DNAm.propeller(seq, overhang = 12)[1] for seq in DNA]

orderedGCContent = sorted(zip(freq, GCContent), key = lambda x: int(x[0]))
orderedRoll = sorted(zip(freq, roll), key = lambda x: int(x[0]))
orderedmGroove = sorted(zip(freq, mGroove), key = lambda x: int(x[0]))
orderedHelixTwist = sorted(zip(freq, helixTwist), key = lambda x: int(x[0]))
orderedPropeller = sorted(zip(freq, propeller), key = lambda x: int(x[0]))

orderedGCContent = [x[1] for x in orderedGCContent]
orderedRoll = [x[1] for x in orderedRoll]
orderedmGroove = [x[1] for x in orderedmGroove]
orderedHelixTwist = [x[1] for x in orderedHelixTwist]
orderedPropeller = [x[1] for x in orderedPropeller]


l = len(freq)

topTenGC = np.mean(orderedGCContent[l - int(l*0.1):l], axis = 0)
bottomTenGC = np.mean(orderedGCContent[0:int(l*0.1)], axis = 0)
topTenRoll = np.mean(orderedRoll[l - int(l*0.1):l], axis = 0)
bottomTenRoll = np.mean(orderedRoll[0:int(l*0.1)], axis = 0)
topTenGroove = np.mean(orderedmGroove[l - int(l*0.1):l], axis = 0)
bottomTenGroove = np.mean(orderedmGroove[0:int(l*0.1)], axis = 0)
topTenHelixTwist = np.mean(orderedHelixTwist[l - int(l*0.1):l], axis = 0)
bottomTenHelixTwist = np.mean(orderedHelixTwist[0:int(l*0.1)], axis = 0)
topTenPropeller = np.mean(orderedPropeller[l - int(l*0.1):l], axis = 0)
bottomTenPropeller = np.mean(orderedPropeller[0:int(l*0.1)], axis = 0)

topTen = [topTenGC, topTenRoll, topTenGroove, topTenHelixTwist, topTenPropeller]
bottomTen = [bottomTenGC, bottomTenRoll, bottomTenGroove, bottomTenHelixTwist, bottomTenPropeller]
ylabels = ['GC content (%)', r'Roll ($\degree$)', r'Minor groove width ($\AA$)',
          r'Helical twist ($\degree$)', r'Propeller twist ($\degree$)']
yScales = [[20,80], [-4,4], [4.5, 5.5], [33, 36], [-12,-2]]
titles = ['Average positional GC content', 'Average positional base pair roll',
          'Average positional minor groove width', 'Average positional base pair helical twist',
          'Average positional base pair propeller twist']

bpRange = range(-12,0)
reversebpRange = range(-12,0)
reversebpRange.reverse()
xlabels = bpRange + ['N1', 'N2', 'N3', 'N4', 'N5'] + reversebpRange
xrange = range(len(xlabels))

fig = pp.figure(figsize = (8, 8))

for i in range(len(topTen)):
    ax = pp.subplot(5,1,i+1)
    ax.plot(topTen[i], label = 'Top 10% insertion rate', marker = '*')
    ax.plot(bottomTen[i], label = 'Bottom 10% insertion rate', marker = '*')
    ax.margins(0.01)
    ax.set_ylim(yScales[i])
    ax.set_title(titles[i])
    ax.set_ylabel(ylabels[i])
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([])

    if i == 0:
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon = False, ncol = 2, loc = "upper right", mode="expand")
    if i == len(topTen) - 1:
        ax.set_xlabel("Nucleotide position")
        ax.set_xticklabels(xlabels)

pp.tight_layout(h_pad = 0)
fig.savefig('DNA Structure Analysis.png')


