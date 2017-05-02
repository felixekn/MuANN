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
            plotType.text(bins.get_x()+bins.get_width()/2., height + offSet*0.01, '%d'%int(height),
                    ha='center', va='bottom')
        else:
            plotType.text(bins.get_x()+bins.get_width()/2., height + offSet*0.01, '%.2f'%(height),
                    ha='center', va='bottom')

# inporting training data
DNA1, recSite1, freq1 = DNAm.array("sixtyninemers_frequencies_GsAK_og.csv")
DNA2, recSite2, freq2 = DNAm.array("sixtyninemers_frequencies_TnAK_og.csv")
DNA3, recSite3, freq3 = DNAm.array("sixtyninemers_frequencies_BgAK_og.csv")
DNA4, recSite4, freq4 = DNAm.array("sixtyninemers_frequencies_BsAK_og.csv")

mGroove1 = [DNAm.minorGroove(seq, overhang = 12)[1] for seq in DNA1]
mGroove2 = [DNAm.minorGroove(seq, overhang = 12)[1] for seq in DNA2]
mGroove3 = [DNAm.minorGroove(seq, overhang = 12)[1] for seq in DNA3]
mGroove4 = [DNAm.minorGroove(seq, overhang = 12)[1] for seq in DNA4]

freq = freq1 + freq2 + freq3 + freq4
mGroove = mGroove1 + mGroove2 + mGroove3 + mGroove4
mGrooveAverageAll = np.mean(mGroove, axis = 0)
mGrooveGeneAverages = [np.mean(mGroove1), np.mean(mGroove2), np.mean(mGroove3), np.mean(mGroove4)]

orderedmGroove = sorted(zip(freq, mGroove), key = lambda x: int(x[0]))
orderedmGroove = [x[1] for x in orderedmGroove]
l = len(orderedmGroove)

averagemGrooveTopTen = np.mean(orderedmGroove[l - int(l*0.1):l], axis = 0)
averagemGrooveBottomTen = np.mean(orderedmGroove[0:int(l*0.1)], axis = 0)

bpRange = range(-12,0)
reversebpRange = range(-12,0)
reversebpRange.reverse()
xlabels = bpRange + ['N1', 'N2', 'N3', 'N4', 'N5'] + reversebpRange

fig = pp.figure(figsize = (12, 7))
ax = pp.subplot2grid((2,4), (0,0), colspan = 4)
ax.plot(mGrooveAverageAll, label = 'Average minor groove width', linewidth = 2, color = 'gray')
ax.plot(averagemGrooveTopTen, label = 'Top 10% insert freq.', marker = '*')
ax.plot(averagemGrooveBottomTen, label = 'Bottom 10% insert freq.', marker = '*')
ax.set_ylim([3.5,6])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title(r"Average positional minor groove width")
ax.set_xlabel("Nucleotide position")
ax.set_ylabel(r"Average minor groove width ($\AA$)")
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)

ax = pp.subplot2grid((2,4), (1,0))
n_bar = len(mGrooveGeneAverages)
ind = np.arange(n_bar)

p1 = ax.bar(ind,mGrooveGeneAverages, width = 0.8)
ax.set_ylim([3.5,6])
ax.set_xticks(range(len(mGrooveGeneAverages)))
ax.set_xticklabels(["GsAK", "TnAK", "BgAK", "BsAK"])
ax.margins(0.05)

ax.set_title("Average gene minor groove width")
ax.set_xlabel("Gene")
ax.set_ylabel(r"Average minor groove width ($\AA$)")
autolabel(p1, ax, integer = False)


ax = pp.subplot2grid((2,4), (1,1), colspan = 3)
ax.plot(mGroove1[92], label = 'High insert freq.', marker = '*')
ax.plot(mGroove1[0], label = 'Low insert freq.', marker = '*')
ax.set_ylim([3.5,6])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Representative low and high freq. minor groove widths")
ax.set_xlabel("Nucleotide position")

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False, loc = 'upper right')


pp.tight_layout()
fig.savefig('Minor Groove Width Analysis.pdf')


