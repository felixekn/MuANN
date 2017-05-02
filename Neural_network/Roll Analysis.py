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
            plotType.text(bins.get_x()+bins.get_width()/2., 0, '-%d'%int(height),
                    ha='center', va='bottom')
        else:
            plotType.text(bins.get_x()+bins.get_width()/2., 0, '-%.2f'%(height),
                    ha='center', va='bottom')

# inporting training data
DNA1, recSite1, freq1 = DNAm.array("sixtyninemers_frequencies_GsAK_og.csv")
DNA2, recSite2, freq2 = DNAm.array("sixtyninemers_frequencies_TnAK_og.csv")
DNA3, recSite3, freq3 = DNAm.array("sixtyninemers_frequencies_BgAK_og.csv")
DNA4, recSite4, freq4 = DNAm.array("sixtyninemers_frequencies_BsAK_og.csv")

roll1 = [DNAm.roll(seq, overhang = 12)[1] for seq in DNA1]
roll2 = [DNAm.roll(seq, overhang = 12)[1] for seq in DNA2]
roll3 = [DNAm.roll(seq, overhang = 12)[1] for seq in DNA3]
roll4 = [DNAm.roll(seq, overhang = 12)[1] for seq in DNA4]

freq = freq1 + freq2 + freq3 + freq4
roll = roll1 + roll2 + roll3 + roll4
rollAverageAll = np.mean(roll, axis = 0)
rollGeneAverages = [np.mean(roll1), np.mean(roll2), np.mean(roll3), np.mean(roll4)]

orderedroll = sorted(zip(freq, roll), key = lambda x: int(x[0]))
orderedroll = [x[1] for x in orderedroll]
l = len(orderedroll)

averagerollTopTen = np.mean(orderedroll[l - int(l*0.1):l], axis = 0)
averagerollBottomTen = np.mean(orderedroll[0:int(l*0.1)], axis = 0)

bpRange = range(-12,0)
reversebpRange = range(-12,0)
reversebpRange.reverse()
xlabels = bpRange + ['N1', 'N2', 'N3', 'N4', 'N5'] + reversebpRange

fig = pp.figure(figsize = (12, 7))
ax = pp.subplot2grid((2,4), (0,0), colspan = 4)
ax.plot(rollAverageAll, label = 'Average roll', linewidth = 2, color = 'gray')
ax.plot(averagerollTopTen, label = 'Top 10% insert freq.', marker = '*')
ax.plot(averagerollBottomTen, label = 'Bottom 10% insert freq.', marker = '*')
ax.set_ylim([-7,7])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title(r"Average positional roll")
ax.set_xlabel("Nucleotide position")
ax.set_ylabel(r"Average roll ($\degree$)")
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)

ax = pp.subplot2grid((2,4), (1,0))
n_bar = len(rollGeneAverages)
ind = np.arange(n_bar)

p1 = ax.bar(ind,rollGeneAverages, width = 0.8)
ax.set_ylim([-7,7])
ax.set_xticks(range(len(rollGeneAverages)))
ax.set_xticklabels(["GsAK", "TnAK", "BgAK", "BsAK"])
ax.margins(0.05)

ax.set_title("Average gene roll")
ax.set_xlabel("Gene")
ax.set_ylabel(r"Average roll ($\degree$)")
autolabel(p1, ax, integer = False)

ax = pp.subplot2grid((2,4), (1,1), colspan = 3)
ax.plot(roll1[91], label = 'High insert freq.', marker = '*')
ax.plot(roll1[0], label = 'Low insert freq.', marker = '*')
ax.set_ylim([-7,7])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Representative low and high freq. roll")
ax.set_xlabel("Nucleotide position")

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False, loc = 'upper right')


pp.tight_layout()
fig.savefig('Roll Analysis.pdf')


