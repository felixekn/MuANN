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
DNA1, recSite1, freq1 = DNAm.array("sixtyninemers_frequencies_GsAK_og.csv")
DNA2, recSite2, freq2 = DNAm.array("sixtyninemers_frequencies_TnAK_og.csv")
DNA3, recSite3, freq3 = DNAm.array("sixtyninemers_frequencies_BgAK_og.csv")
DNA4, recSite4, freq4 = DNAm.array("sixtyninemers_frequencies_BsAK_og.csv")

GCContent1 = [DNAm.GC_content(seq, overhang = 12)[1] for seq in DNA1]
GCContent2 = [DNAm.GC_content(seq, overhang = 12)[1] for seq in DNA2]
GCContent3 = [DNAm.GC_content(seq, overhang = 12)[1] for seq in DNA3]
GCContent4 = [DNAm.GC_content(seq, overhang = 12)[1] for seq in DNA4]

freq = freq1 + freq2 + freq3 + freq4
GCContent = GCContent1 + GCContent2 + GCContent3 + GCContent4
GCAverageAll = np.mean(GCContent, axis = 0)
GCGeneAverages = [np.mean(GCContent1), np.mean(GCContent2), np.mean(GCContent3), np.mean(GCContent4)]

orderedGCContent = sorted(zip(freq, GCContent), key = lambda x: int(x[0]))
orderedGCContent = [x[1] for x in orderedGCContent]
l = len(orderedGCContent)

averageGCTopTen = np.mean(orderedGCContent[l - int(l*0.1):l], axis = 0)
averageGCBottomTen = np.mean(orderedGCContent[0:int(l*0.1)], axis = 0)

bpRange = range(-12,0)
reversebpRange = range(-12,0)
reversebpRange.reverse()
xlabels = bpRange + ['N1', 'N2', 'N3', 'N4', 'N5'] + reversebpRange
xrange = range(len(xlabels))

fig = pp.figure(figsize = (12, 7))
ax = pp.subplot2grid((2,4), (0,0), colspan = 4)
ax.plot(GCAverageAll, label = 'Average GC', linewidth = 2, color = 'gray')
ax.plot(averageGCTopTen,label = 'Top 10% insert freq.', marker = '*')
ax.plot(averageGCBottomTen, label = 'Bottom 10% insert freq.', marker = '*')
ax.set_ylim([0,100])
ax.set_xticks(xrange)
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Average positional GC content")
ax.set_xlabel("Nucleotide position")
ax.set_ylabel("Average GC content (%)")
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)

ax = pp.subplot2grid((2,4), (1,0))
n_bar = len(GCGeneAverages)
ind = np.arange(n_bar)

p1 = ax.bar(ind,GCGeneAverages, width = 0.8)
ax.set_ylim([0,100])
ax.set_xticks(range(len(GCGeneAverages)))
ax.set_xticklabels(["GsAK", "TnAK", "BgAK", "BsAK"])
ax.margins(0.05)

ax.set_title("Average gene GC content")
ax.set_xlabel("Gene")
ax.set_ylabel("Average GC content (%)")
autolabel(p1, ax, integer = False)


ax = pp.subplot2grid((2,4), (1,1), colspan = 3)
ax.plot(GCContent1[92], label = 'High insert freq.', marker = '*')
ax.plot(GCContent1[0], label = 'Low insert freq.', marker = '*')
ax.set_ylim([0,100])
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.margins(0.01)
ax.set_title("Representative low and high freq GC content")
ax.set_xlabel("Nucleotide position")

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)


pp.tight_layout()
fig.savefig('GC Analysis.pdf')


