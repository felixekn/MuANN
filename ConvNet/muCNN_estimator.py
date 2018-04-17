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

# CNN parameters

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

# Training CNN
print("Training neural net...")
weightsTraining, totalStepsTraining, errorTraining = network.BP_learn(scaledTrainingInput, scaledTrainingOutput, scaledTestInput, scaledTestOutput, networkLayers, slope, eta, n, tol, weights, batch, alpha, decay)

# Recall output from trained ANN
print("Recalling training data from neural net...")
recallTraining = network.BP_recall(weightsTraining, scaledTrainingInput, slope)
thresholdedRecall = network.threshold(recallTraining)
binnedRecallTraining = network.categoriesToBins(thresholdedRecall, nbins = 10)

