from matplotlib import pyplot as pp
import numpy as np
import Neural_network.DNA_manipulations as DNAm
import Neural_network.Neural_network as network

# --------------------------------------------------------- #
# --------------------------------------------------------- #
# ------------------  Main Script Below ------------------- #
# ------------------ For CLASSIFICATION ------------------- #
# --------------------------------------------------------- #
# --------------------------------------------------------- #

# ANN parameters
networkLayers = [63,40,4]
slope = 1    # tanh slope
eta = 0.025   # learning rate
n = 50000     # learning steps
tol = 0.99  # hit rate
batch = False # batch training, if false = inline training
alpha = 0.7 # influence of momentum
decay = 0   # weight decay term
weights = network.initialize_weights(networkLayers, 0.5)

# inporting training data
print('Importing Data...')
DNA_GsAK, recSite_GsAK, freq_GsAK = DNAm.array("Data/GsAK_unselected_NucleotidePositionCounts.csv")
DNA_TnAK, recSite_TnAK, freq_TnAK = DNAm.array("Data/TnAK_unselected_NucleotidePositionCounts.csv")
DNA_BsAK, recSite_BsAK, freq_BsAK = DNAm.array("Data/BsAK_unselected_NucleotidePositionCounts.csv")
DNA_BgAK, recSite_BgAK, freq_BgAK = DNAm.array("Data/BgAK_unselected_NucleotidePositionCounts.csv")

# Training and test outputs
print('Splitting data into Training/Test...')
freq = np.array([[x] for x in freq_GsAK + freq_BsAK + freq_BgAK])
freqTest = np.array([[x] for x in freq_TnAK])

#Scale data
scaledTrainingOutput, binendTainingOutput = network.categorize(freq, 0, 15000, 10)
scaledTestOutput, binendTestOutput = network.categorize(freqTest, 0, 15000, 10)

# Training and test input data
DNA = DNA_GsAK + DNA_BsAK + DNA_BgAK
DNATest = DNA_TnAK

print('Encoding data...')

print('\t One-hot sequence')
encodedRecSiteTraining = [DNAm.DNA_encoding(seq, overhang = 0) for seq in DNA]
encodedRecSiteTest = [DNAm.DNA_encoding(seq, overhang = 0) for seq in DNATest]
print(np.shape(np.array(encodedRecSiteTraining[1])))
print('\t GC content')
GC_averageTraining = [DNAm.GC_content(seq, overhang = 7)[2] for seq in DNA]
GC_averageTest = [DNAm.GC_content(seq, overhang = 7)[2] for seq in DNATest]

print('\t Propeller twist')
propellerTraining = [DNAm.propeller(seq, overhang = 7)[2] for seq in DNA]
propellerTest = [DNAm.propeller(seq, overhang = 7)[2] for seq in DNATest]

print('\t Roll')
rollTraining = [DNAm.roll(seq, overhang = 0)[2] for seq in DNA]
rollTest = [DNAm.roll(seq, overhang = 0)[2] for seq in DNATest]

print('\t Helix twist')
helixTraining = [DNAm.helixTwist(seq, overhang = 0)[2] for seq in DNA]
helixTest = [DNAm.helixTwist(seq, overhang = 0)[2] for seq in DNATest]

print('\t Groove')
grooveTraining = [DNAm.minorGroove(seq, overhang = 0)[2] for seq in DNA]
grooveTest = [DNAm.minorGroove(seq, overhang = 0)[2] for seq in DNATest]

# concatenating input training data
print(np.shape(np.array(encodedRecSiteTraining)))
print(np.shape(np.array(GC_averageTraining)))
print(np.shape(np.array(propellerTraining)))
print(np.shape(np.array(rollTraining)))
print(np.shape(np.array(helixTraining)))
print(np.shape(np.array(grooveTraining)))

inputData = np.column_stack((GC_averageTraining, propellerTraining,
                             rollTraining, helixTraining, grooveTraining,
                             encodedRecSiteTraining))
testData = np.column_stack((GC_averageTest, propellerTest, rollTest,
                            helixTest, grooveTest, encodedRecSiteTest))

# scaling input and output training data
scaledTrainingInput = inputData
scaledTestInput = testData

# Training ANN
print("Training neural net...")
weightsTraining, totalStepsTraining, errorTraining = network.BP_learn(scaledTrainingInput, scaledTrainingOutput, scaledTestInput, scaledTestOutput, networkLayers, slope, eta, n, tol, weights, batch, alpha, decay)

# Recall output from trained ANN
print("Recalling training data from neural net...")
recallTraining = network.BP_recall(weightsTraining, scaledTrainingInput, slope)
thresholdedRecall = network.threshold(recallTraining)
binnedRecallTraining = network.categoriesToBins(thresholdedRecall, nbins = 10)

# Recall output from trained ANN
print("Recalling test data from neural net...")
recallTest = network.BP_recall(weightsTraining, scaledTestInput, slope)
thresholdedTestRecall = network.threshold(recallTest)
binnedRecallTest = network.categoriesToBins(thresholdedTestRecall, nbins = 10)

# printing hit rate
print("Training recall hit rate percent: %0.2f"%(network.percent_identity(scaledTrainingOutput, thresholdedRecall)))
print("Test recall hit rate percent: %0.2f"%(network.percent_identity(scaledTestOutput, thresholdedTestRecall)))


# Plotting Training and Test Inputs and their Recalled Outputs
figWidth = 4.2
figHeight = 4
fig = pp.figure(figsize=(figWidth, figHeight))
ax = pp.subplot()
ax.plot(totalStepsTraining, errorTraining[0], label = 'Training recall', color = [0, 0.4470, 0.7410])
ax.plot(totalStepsTraining, errorTraining[1], label = 'Test recall', color = "#8E44AD")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon = False)
ax.set_title("Training history")
ax.set_ylabel("Hit rate (% correct)")
ax.set_xlabel('Learning step')
pp.tight_layout()
pp.savefig('Training Histroy (categorize).pdf')


figWidth = 14
figHeight = 8
fig = pp.figure(figsize=(figWidth, figHeight))
ax2 = pp.subplot(2,3,1)
ax2.bar(range(len(binendTainingOutput)), binendTainingOutput, width = 1)
ax2.bar([85], binendTainingOutput[85], width = 1,  color = '#F9690E')
ax2.set_title("Scaled training Mu insertion frequencies")
ax2.set_ylabel("Scaled insertion rate (au)")
ax2.set_xlabel('Unique sequence')

ax3 = pp.subplot(2,3,2)
ax3.bar(range(len(binnedRecallTraining)), binnedRecallTraining, width = 1)
ax3.bar([85], binnedRecallTraining[85], width = 1,  color = '#F9690E')
ax3.set_title("Recalled scaled training Mu insertion frequencies")
ax3.set_ylabel("Scaled insertion rate (au)")
ax3.set_xlabel('Unique sequence')
ax3.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]])

ax4 = pp.subplot(2,3,4)
ax4.bar(range(len(binendTestOutput)), binendTestOutput, width = 1, color = "#8E44AD")
ax4.bar([97], binendTestOutput[97], width = 1, color = '#F9690E')
ax4.set_ylabel("Scaled insertion rate (au)")
ax4.set_xlabel('Unique sequence')
ax4.set_title("Scaled test Mu insertion frequencies")

ax5 = pp.subplot(2,3,5)
ax5.bar(range(len(binnedRecallTest)), binnedRecallTest, width = 1, color = "#8E44AD")
ax5.bar([97], binnedRecallTest[97], width = 1,  color = '#F9690E')
ax5.set_title("Recalled scaled test Mu insertion frequencies")
ax5.set_ylabel("Scaled insertion rate (au)")
ax5.set_xlabel('Unique sequence')
ax5.set_ylim([ax4.get_ylim()[0], ax4.get_ylim()[1]])

ax = pp.subplot(2,3,3)
ax.plot(binendTainingOutput, binnedRecallTraining, linestyle = 'none', marker = 'o', markerfacecolor = [37/255., 116/255., 169/255., 0.05])
ax.set_title("Actual vs recall output: training data ")
ax.set_ylabel("Desired insertion rate (au)")
ax.set_xlabel("Recalled insertion rate (au)")

ax = pp.subplot(2,3,6)
ax.plot(binendTestOutput, binnedRecallTest, linestyle = 'none', marker = 'o', markerfacecolor = [142/255., 68/255., 173/255., 0.05], color = "#8E44AD")
ax.set_title("Actual vs recall output: test data ")
ax.set_ylabel("Desired insertion rate (au)")
ax.set_xlabel("Recalled insertion rate (au)")

pp.tight_layout()
fig.savefig('training (categorize).pdf')






