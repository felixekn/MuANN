import csv
import numpy as np
import copy

def array(file):
	"""
	Imports a csv of DNA sequences and inserts them into an array
	"""
	sequences = []
	recSite = []
	freq = []
	with open(file, 'r') as csv_file:
		fileReader = csv.reader(csv_file, delimiter = "|")
		fileReader.next() # throwaway header row

	  	for row in fileReader:
			strippedRow = row[0].strip(",").split(',')
			sequences.append(strippedRow[1])
			recSite.append(strippedRow[2])
			freq.append(int(strippedRow[4]))

	return sequences, recSite, freq


def GC_content(sequence, recLength = 5, overhang = 12, window = 3,
			   ymax = 1, ymin = -1):
	"""
	Takes a DNA sequence and calculates the running average GC content
	with a defualt window of 3 basepairs over the length of the sequence
	"""
	GC_array = []
	maxGC = 100
	minGC = 0

	# GC percentages
	for bp in sequence:
		if bp.capitalize() in ['G', 'C']:
			GC_array.append(100)
		else:
			GC_array.append(0)

	# window weighting
	weights = np.repeat(1.0, window)/float(window)
	runningAverage = np.convolve(GC_array, weights, 'valid')

	# normalizing data
	normalize = (ymax - ymin)*(runningAverage - minGC)/(maxGC - minGC) + ymin;

	# pulling out feature indecies
	middle = len(runningAverage)/2 + len(runningAverage)%2 - 1
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	return GC_array, runningAverage[start-overhang:end+overhang], normalize[start-overhang:end+overhang]



def propeller(sequence, recLength = 5, overhang = 12, window = 3,
			  ymax = 1, ymin = -1):
	"""
	Takes a string of DNA in and converts it to its propeller twist
	"""
	twistMapping = {'AAATG': -11.76, 'GCCCG': -2.73, 'AAATC': -12.27, 'AAATA': -12.63, 'AAATT': -14.95, 'GCGTG': -7.98, 'TGGTG': -7.0, 'GGCTG': -1.2, 'TGGTC': -6.17, 'GTGGC': -4.03, 'AGACA': -8.49, 'GATAG': -8.88, 'AGACC': -8.04, 'GATAC': -8.06, 'AGACG': -8.81, 'AAGTT': -6.26, 'TTACG': -7.71, 'TTACA': -7.55, 'AGACT': -7.81, 'TTACC': -7.21, 'AGCAA': -3.36, 'TCGCC': -2.99, 'AGCAC': -3.79, 'TCGCG': -4.27, 'AGCAG': -3.6, 'TTCAG': -5.95, 'TTCAC': -6.21, 'TCGGG': -2.92, 'GAAAG': -11.93, 'TCATG': -9.26, 'GAAAC': -12.76, 'TCATC': -7.37, 'AACAG': -9.26, 'AACAC': -9.4, 'AACAA': -10.17, 'TGTTG': -11.69, 'TGTTC': -10.91, 'GACTG': -3.96, 'TATAG': -7.69, 'AACAT': -10.22, 'TATAA': -7.14, 'TATAC': -7.88, 'GTCGG': -6.74, 'CTAGG': -6.51, 'TAATG': -10.52, 'TGGCG': -2.98, 'TAATC': -10.25, 'TGGCC': -2.54, 'TAATA': -9.98, 'TGGCA': -2.98, 'GCTTG': -8.63, 'AATGT': -10.82, 'CGGCG': -3.17, 'AATGG': -8.98, 'GTGGG': -2.94, 'AATGC': -9.59, 'AATGA': -9.02, 'ATACA': -8.93, 'ATACC': -8.36, 'ATACG': -9.05, 'ACTAC': -8.16, 'GGGCG': -2.89, 'ACTAA': -8.1, 'ACTAG': -7.44, 'GGGCC': -1.93, 'TGACG': -8.47, 'ATACT': -8.21, 'TGACA': -8.28, 'TGACC': -7.96, 'GCCGG': -3.58, 'AGTTA': -10.74, 'AGTTC': -10.45, 'AGTTG': -11.29, 'AGGTC': -5.67, 'AGGTA': -5.43, 'AGGTG': -6.64, 'TAACG': -11.31, 'TAACC': -10.83, 'TAACA': -11.35, 'GTGCC': -3.41, 'GTGCG': -4.13, 'CACAG': -8.12, 'AACGC': -9.91, 'GTGTG': -8.16, 'GCTCG': -6.78, 'ACATC': -8.65, 'ATGAT': -6.15, 'ACATG': -10.15, 'TCTTC': -8.18, 'TCTTG': -9.12, 'ATGAA': -6.07, 'ATGAC': -6.37, 'TATGA': -7.0, 'ATGAG': -6.05, 'ACCCC': -2.37, 'ACCCA': -3.38, 'CTGCG': -3.85, 'ACCCG': -3.53, 'GGTGG': -7.11, 'GGTGC': -7.89, 'ACTTG': -10.21, 'TAGCG': -1.63, 'ACTTA': -9.96, 'TAGCA': -1.6, 'ACTTC': -8.83, 'TAGCC': -1.22, 'TGAGA': -6.62, 'TGAGC': -6.75, 'TGAGG': -6.36, 'ATCTG': -2.7, 'GTAAC': -10.83, 'ATCTC': -2.78, 'ATCTA': -3.2, 'GTAAG': -10.31, 'CAACG': -11.71, 'GGAGC': -6.07, 'AGGAG': -4.81, 'GGAGG': -5.75, 'AGGAC': -5.39, 'TCTAC': -6.38, 'TGGGC': -2.77, 'TGGGA': -2.42, 'TCTAG': -6.98, 'TGGGG': -2.02, 'ATGTC': -7.1, 'ATGTA': -7.4, 'ATGTG': -8.83, 'ACAAC': -11.7, 'ATTCT': -10.15, 'ACAAA': -12.03, 'AAGTC': -4.07, 'ACAAG': -10.72, 'TTAGG': -5.77, 'AAGTG': -4.48, 'TCCAC': -3.48, 'AATCT': -10.31, 'GAGCC': -0.53, 'TTGCG': -4.14, 'ATTCG': -10.56, 'ATTCA': -10.83, 'ATTCC': -9.61, 'TGCCC': -2.52, 'AGGAA': -4.88, 'AATCA': -9.91, 'AATCG': -9.51, 'ATTGC': -10.6, 'TTGGA': -2.74, 'TTGGG': -3.45, 'TGCGG': -4.13, 'TGCGC': -3.94, 'TGCGA': -3.81, 'GGGGC': -2.13, 'GTCTG': -3.97, 'TAGGA': -1.39, 'TAGGC': -1.23, 'TAGGG': -0.56, 'AGATG': -7.85, 'AGATA': -8.23, 'AGATC': -8.23, 'TAAGC': -8.92, 'TAAGA': -8.51, 'TAAGG': -8.13, 'TGGAG': -5.45, 'AAGAC': -3.82, 'AAGAA': -3.46, 'AAGAG': -3.22, 'GCTGG': -5.87, 'TCACC': -7.04, 'AAGAT': -3.55, 'CTGGG': -2.47, 'ACCGG': -5.03, 'ACCGC': -4.55, 'GGTCG': -8.09, 'ACCGA': -4.29, 'TTAGA': -6.61, 'AAGTA': -4.26, 'TGTAG': -8.6, 'TGTAC': -8.0, 'TTAGC': -6.12, 'TACAC': -6.58, 'TACAA': -7.17, 'TACAG': -6.49, 'TTGCA': -3.8, 'AGTCG': -8.15, 'AGTCA': -8.11, 'CAAGG': -9.16, 'AGTCC': -7.34, 'GCGAG': -6.31, 'CATAG': -8.2, 'GGACG': -8.11, 'GGACC': -7.55, 'TGAAC': -10.38, 'AACGT': -10.39, 'ATTGT': -11.94, 'GAAGG': -7.38, 'GTTAG': -10.65, 'GAAGC': -8.01, 'ATTGA': -10.92, 'AACGG': -10.04, 'AACGA': -9.62, 'ATTGG': -10.72, 'GAGGG': -0.03, 'ATCCT': -5.29, 'GAGGC': -0.7, 'ATCCG': -4.49, 'TCCAG': -2.98, 'GTATG': -8.42, 'ATCCC': -4.91, 'ATCCA': -5.7, 'ACGAA': -6.66, 'GAGCG': -1.32, 'ACGAC': -6.61, 'TTGCC': -3.14, 'TGCCG': -3.11, 'ACGAG': -6.5, 'GTCCG': -6.07, 'GTCCC': -5.28, 'AATCC': -9.67, 'GGCAG': -3.2, 'TACTG': -4.2, 'TACTC': -3.05, 'CCGGG': -3.26, 'AGCGT': -3.93, 'AGCGC': -3.87, 'AGCGA': -3.56, 'AGCGG': -3.78, 'GATTG': -10.33, 'TTTAC': -11.89, 'CAGGG': -0.14, 'TTTAG': -11.18, 'TCAGG': -5.31, 'CTTCG': -9.8, 'TCAGA': -6.32, 'TCAGC': -6.36, 'AAAAT': -14.89, 'TTGGC': -3.85, 'GACAC': -7.25, 'GACAG': -6.88, 'AAAAA': -16.51, 'TTGTG': -8.23, 'AAAAC': -14.47, 'TTGTC': -7.48, 'AAAAG': -14.68, 'GGATG': -7.65, 'GGGGG': -1.03, 'AGTGA': -7.1, 'AGTGC': -7.68, 'AGTGG': -7.4, 'TTTTG': -13.79, 'TTTTC': -13.16, 'AGTGT': -8.48, 'ATGGG': -2.37, 'ATGGC': -3.12, 'ATGGA': -3.37, 'CCAGG': -5.54, 'ATGGT': -4.33, 'GAACC': -10.08, 'GAACG': -11.06, 'AACCA': -8.86, 'AACCC': -7.68, 'AACCG': -7.97, 'ATCGT': -6.85, 'AACCT': -8.15, 'ATCGC': -6.96, 'ATCGA': -7.08, 'ATCGG': -7.15, 'GTAGG': -6.19, 'GTAGC': -6.96, 'GCCTG': -1.28, 'CTAAG': -10.31, 'GTCGC': -6.9, 'GGGTG': -5.89, 'GCACG': -8.4, 'TTATC': -7.66, 'TTATG': -8.06, 'AGAAA': -10.55, 'ACCTC': -0.88, 'ACCTA': -1.54, 'ACCTG': -1.68, 'CAGCG': -1.72, 'ATAAC': -10.51, 'ATAAA': -12.37, 'ATAAG': -10.54, 'TCACG': -8.08, 'CTTGG': -9.71, 'GAGTC': -3.11, 'GAGTG': -3.31, 'TGAAG': -10.24, 'ACTGG': -7.24, 'ACTGA': -7.1, 'ATAAT': -11.27, 'ACTGC': -7.79, 'GCCAG': -3.34, 'TGGAC': -5.78, 'GTGAC': -6.82, 'GTGAG': -6.5, 'AGAAG': -10.3, 'GATCG': -7.79, 'AGAAC': -10.59, 'CGCGG': -4.25, 'GATCC': -7.56, 'TTAAC': -10.44, 'TTAAA': -10.7, 'TTAAG': -9.57, 'TCGAC': -6.68, 'AGCCG': -3.04, 'ATGCC': -3.12, 'ATGCA': -3.73, 'AGCCC': -2.48, 'ATGCG': -4.03, 'AGCCA': -2.76, 'TCGAG': -6.72, 'TTCCC': -4.39, 'CTGAG': -6.08, 'TTCCA': -5.2, 'TTCCG': -5.36, 'CAGTG': -4.52, 'ACCAG': -4.44, 'ACCAA': -4.69, 'ACCAC': -5.0, 'TAGAG': -3.22, 'TAGAC': -3.62, 'TAGAA': -3.73, 'TATCC': -7.47, 'TATCA': -7.32, 'CTCCG': -5.07, 'TATCG': -7.63, 'TGCTC': -0.94, 'ACGTG': -8.78, 'TCCTG': -0.9, 'GTACC': -8.12, 'TCCTC': -0.44, 'GTACG': -8.84, 'TGCTG': -1.72, 'GCGGG': -2.75, 'GCGGC': -3.73, 'GCAGG': -5.92, 'AGCTG': -1.38, 'TCTCG': -6.8, 'ACGTA': -7.92, 'GCAGC': -6.76, 'AGCTC': -0.83, 'TCTCC': -6.1, 'AGCTA': -1.3, 'ATGCT': -3.68, 'GTTGC': -10.75, 'ACACG': -9.43, 'ACACA': -9.33, 'ACACC': -8.57, 'ATATG': -8.33, 'ATATA': -8.95, 'ATATC': -8.3, 'GAGAG': -2.54, 'GAGAC': -3.46, 'GAATG': -9.43, 'GAATC': -9.46, 'ATTTA': -13.87, 'ATTTC': -12.09, 'GGGAG': -4.6, 'ATTTG': -13.36, 'ACTCA': -7.5, 'TGTGG': -7.33, 'ACTCC': -6.75, 'TGTGA': -7.82, 'TGTGC': -8.3, 'ACTCG': -7.78, 'TATGG': -6.69, 'GGCGG': -3.13, 'TATGC': -8.11, 'GGCGC': -3.39, 'CAATG': -10.42, 'TAAAG': -12.85, 'TAAAA': -13.29, 'TAAAC': -13.54, 'CACCG': -6.56, 'GATGC': -7.6, 'GATGG': -7.48, 'AAGCG': -1.91, 'AAGCA': -1.47, 'AAGCC': -1.55, 'GCTAG': -6.54, 'AAGCT': -1.56, 'AACTG': -5.42, 'AACTA': -6.07, 'AACTC': -5.23, 'TTCGG': -6.08, 'GTTCG': -10.48, 'TTCGC': -6.08, 'TTCGA': -5.99, 'CGTGG': -7.87, 'GTTCC': -9.91, 'TGTCC': -7.71, 'TGTCG': -8.25, 'TACGG': -6.7, 'TACGA': -6.48, 'TACGC': -7.22, 'AGTAG': -8.16, 'CAAAG': -12.85, 'AGTAC': -8.16, 'AGTAA': -7.55, 'AGGCC': -2.2, 'AGGCA': -2.38, 'AGGCG': -2.82, 'GCGCG': -4.22, 'CATGG': -9.01, 'GGAAG': -9.44, 'TCTGC': -6.69, 'TCTGG': -6.09, 'AGGCT': -2.34, 'ACAGA': -7.0, 'ACAGC': -7.34, 'ACAGG': -6.78, 'ATTAC': -11.34, 'ATTAA': -11.32, 'ATTAG': -11.58, 'ACAGT': -8.12, 'AATAA': -9.65, 'AATAC': -10.45, 'AATAG': -11.61, 'TCCCG': -2.56, 'ACGGT': -4.79, 'TCCCC': -1.44, 'ACGGC': -3.97, 'ACGGA': -3.36, 'CTCGG': -6.12, 'AATAT': -10.43, 'TTGAG': -6.36, 'GTCAG': -6.43, 'TGATC': -7.81, 'TGATG': -8.13, 'GGCCG': -2.72, 'TTGAA': -5.98, 'TTGAC': -7.07, 'GTTTG': -13.1, 'ACGGG': -2.81, 'TCGGC': -3.4, 'TCGGA': -3.04, 'CGACG': -8.49, 'AAGGA': -1.33, 'AAGGC': -1.5, 'TTTGG': -9.32, 'TTTGA': -10.28, 'AAGGG': -0.14, 'TTTGC': -10.88, 'AAGGT': -1.48, 'AATTC': -11.18, 'AATTA': -12.2, 'AATTG': -11.93, 'AAAGT': -11.68, 'GACGC': -7.38, 'GACGG': -7.15, 'AAAGC': -10.58, 'TCGTC': -7.11, 'AAAGA': -10.71, 'AAAGG': -9.21, 'GGTAG': -8.07, 'ACGCG': -4.73, 'TGCAC': -3.74, 'ACGCC': -3.63, 'ACGCA': -4.25, 'TGCAG': -3.68, 'TAGTA': -3.97, 'TCGTG': -7.95, 'TAGTC': -3.9, 'TAGTG': -4.24, 'GCATG': -9.43, 'TACCA': -5.96, 'TACCC': -5.01, 'TACCG': -5.99, 'AGAGG': -6.38, 'AGGGT': -2.62, 'AGAGA': -6.62, 'AGAGC': -6.5, 'AGGGG': -1.32, 'AGAGT': -7.83, 'AGGGC': -1.88, 'AGGGA': -1.76, 'CATCG': -8.32, 'CACGG': -8.46, 'TTCTC': -2.4, 'TTCTG': -3.52, 'TCCGC': -3.15, 'TCCGG': -3.19, 'ATCAG': -6.55, 'ATCAA': -7.12, 'ATCAC': -6.53, 'AAACG': -13.15, 'CTACG': -8.43, 'AAACC': -12.41, 'ACGTC': -7.92, 'AAACA': -13.05, 'ACATA': -8.52, 'AAACT': -12.78, 'GTTGG': -9.8, 'GCAAG': -9.9, 'CGGGG': -1.95, 'CGAGG': -6.32, 'TTTCA': -10.81, 'TTTCC': -10.09, 'GGTTG': -4.14, 'CAGAG': -3.06, 'TTTCG': -10.58, 'TCAAC': -10.22, 'TCAAG': -9.98, 'ATAGG': -6.87, 'ATAGA': -7.03, 'ATAGC': -6.84, 'ATAGT': -8.26, 'TATTC': -10.01, 'TATTG': -9.7, 'GACCG': -5.99, 'GACCC': -5.33}
	complimentMatrix = {'A':'T', 'T':'A',
				  		'G':'C', 'C':'G'}

	maxTwist = max(twistMapping.values())
	minTwist = min(twistMapping.values())

	ptwist = []
	DNAarray = list(sequence.upper())

	for index in range(len(DNAarray)-4):
		pentamer = ''.join(DNAarray[index:index+5])

		if pentamer in twistMapping.keys():
			ptwist.append(twistMapping[pentamer])

		else:
			pentamerCompliment = ''
			for bp in reversed(DNAarray[index:index+5]):
				pentamerCompliment = pentamerCompliment + complimentMatrix[bp]
			ptwist.append(twistMapping[pentamerCompliment])

	# running average of window = 3
	weights = np.repeat(1.0, window)/float(window)
	runningAverage = np.convolve(ptwist, weights, 'valid')

	# normalizing data
	normalize = (ymax - ymin)*(runningAverage - minTwist)/(maxTwist - minTwist) + ymin;

	# pulling out feature indecies
	middle = len(runningAverage)/2 + len(runningAverage)%2 - 1
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	return ptwist, runningAverage[start-overhang:end+overhang], normalize[start-overhang:end+overhang]


def minorGroove(sequence, recLength = 5, overhang = 12, window = 3,
				ymax = 1, ymin = -1):
	"""
	Takes a string of DNA in and converts it to the minor groove widths
	"""
	grooveMapping = {'AAATG': 3.84, 'GCCCG': 4.95, 'AAATC': 4.12, 'AAATA': 3.79, 'AAATT': 2.85, 'GCGTG': 5.25, 'TGGTG': 4.84, 'GGCTG': 4.95, 'TGGTC': 4.84, 'GTGGC': 5.46, 'AGACA': 4.99, 'GATAG': 5.54, 'AGACC': 4.73, 'GATAC': 5.65, 'AGACG': 4.83, 'AAGTT': 3.34, 'TTACG': 6.11, 'TTACA': 5.99, 'AGACT': 4.57, 'TTACC': 5.43, 'AGCAA': 5.08, 'TCGCC': 5.4, 'AGCAC': 5.43, 'TCGCG': 5.42, 'AGCAG': 5.16, 'TTCAG': 5.93, 'TTCAC': 5.94, 'TCGGG': 5.22, 'GAAAG': 4.36, 'TCATG': 5.27, 'GAAAC': 4.74, 'TCATC': 5.08, 'AACAG': 4.95, 'AACAC': 5.05, 'AACAA': 4.97, 'TGTTG': 5.24, 'TGTTC': 4.85, 'GACTG': 5.0, 'TATAG': 5.79, 'AACAT': 4.3, 'TATAA': 6.07, 'TATAC': 6.01, 'GTCGG': 5.54, 'CTAGG': 5.69, 'TAATG': 4.9, 'TGGCG': 5.06, 'TAATC': 4.81, 'TGGCC': 4.93, 'TAATA': 5.11, 'TGGCA': 5.24, 'GCTTG': 4.94, 'AATGT': 4.27, 'CGGCG': 5.09, 'AATGG': 5.08, 'GTGGG': 5.41, 'AATGC': 4.89, 'AATGA': 4.46, 'ATACA': 5.82, 'ATACC': 5.46, 'ATACG': 5.53, 'ACTAC': 5.62, 'GGGCG': 4.96, 'ACTAA': 5.73, 'ACTAG': 5.13, 'GGGCC': 4.77, 'TGACG': 5.39, 'ATACT': 5.37, 'TGACA': 5.43, 'TGACC': 4.94, 'GCCGG': 5.36, 'AGTTA': 4.67, 'AGTTC': 4.25, 'AGTTG': 4.38, 'AGGTC': 4.18, 'AGGTA': 4.33, 'AGGTG': 4.42, 'TAACG': 5.15, 'TAACC': 4.87, 'TAACA': 5.17, 'GTGCC': 5.66, 'GTGCG': 5.88, 'CACAG': 5.32, 'AACGC': 4.64, 'GTGTG': 5.73, 'GCTCG': 5.2, 'ACATC': 4.82, 'ATGAT': 4.79, 'ACATG': 4.99, 'TCTTC': 4.52, 'TCTTG': 4.92, 'ATGAA': 5.5, 'ATGAC': 5.45, 'TATGA': 5.84, 'ATGAG': 5.23, 'ACCCC': 4.58, 'ACCCA': 4.66, 'CTGCG': 5.67, 'ACCCG': 4.7, 'GGTGG': 5.56, 'GGTGC': 5.56, 'ACTTG': 4.52, 'TAGCG': 5.22, 'ACTTA': 4.48, 'TAGCA': 5.36, 'ACTTC': 4.31, 'TAGCC': 5.0, 'TGAGA': 5.29, 'TGAGC': 5.4, 'TGAGG': 5.3, 'ATCTG': 4.98, 'GTAAC': 5.86, 'ATCTC': 4.74, 'ATCTA': 4.74, 'GTAAG': 5.62, 'CAACG': 4.98, 'GGAGC': 5.05, 'AGGAG': 4.63, 'GGAGG': 4.93, 'AGGAC': 4.81, 'TCTAC': 5.76, 'TGGGC': 4.96, 'TGGGA': 4.93, 'TCTAG': 5.84, 'TGGGG': 4.89, 'ATGTC': 5.02, 'ATGTA': 5.27, 'ATGTG': 5.15, 'ACAAC': 5.58, 'ATTCT': 4.43, 'ACAAA': 5.21, 'AAGTC': 3.74, 'ACAAG': 5.19, 'TTAGG': 5.96, 'AAGTG': 4.14, 'TCCAC': 5.58, 'AATCT': 3.75, 'GAGCC': 4.67, 'TTGCG': 5.81, 'ATTCG': 4.88, 'ATTCA': 5.22, 'ATTCC': 4.69, 'TGCCC': 5.02, 'AGGAA': 4.76, 'AATCA': 4.46, 'AATCG': 4.56, 'ATTGC': 5.15, 'TTGGA': 5.42, 'TTGGG': 5.44, 'TGCGG': 5.59, 'TGCGC': 5.71, 'TGCGA': 5.68, 'GGGGC': 4.82, 'GTCTG': 5.22, 'TAGGA': 4.98, 'TAGGC': 4.97, 'TAGGG': 4.99, 'AGATG': 4.66, 'AGATA': 4.72, 'AGATC': 4.36, 'TAAGC': 5.01, 'TAAGA': 4.93, 'TAAGG': 4.81, 'TGGAG': 5.02, 'AAGAC': 4.65, 'AAGAA': 4.8, 'AAGAG': 4.68, 'GCTGG': 5.53, 'TCACC': 5.4, 'AAGAT': 3.9, 'CTGGG': 5.39, 'ACCGG': 5.16, 'ACCGC': 5.08, 'GGTCG': 4.82, 'ACCGA': 5.02, 'TTAGA': 6.0, 'AAGTA': 4.03, 'TGTAG': 5.74, 'TGTAC': 6.2, 'TTAGC': 5.82, 'TACAC': 5.74, 'TACAA': 5.89, 'TACAG': 5.41, 'TTGCA': 5.78, 'AGTCG': 4.59, 'AGTCA': 4.89, 'CAAGG': 4.65, 'AGTCC': 4.51, 'GCGAG': 5.46, 'CATAG': 5.53, 'GGACG': 5.02, 'GGACC': 4.8, 'TGAAC': 5.35, 'AACGT': 4.21, 'ATTGT': 5.03, 'GAAGG': 4.63, 'GTTAG': 5.8, 'GAAGC': 4.82, 'ATTGA': 5.46, 'AACGG': 4.62, 'AACGA': 4.8, 'ATTGG': 5.3, 'GAGGG': 4.78, 'ATCCT': 4.51, 'GAGGC': 4.85, 'ATCCG': 4.47, 'TCCAG': 5.31, 'GTATG': 5.93, 'ATCCC': 4.81, 'ATCCA': 4.94, 'ACGAA': 5.21, 'GAGCG': 5.02, 'ACGAC': 5.34, 'TTGCC': 5.3, 'TGCCG': 5.17, 'ACGAG': 5.13, 'GTCCG': 5.08, 'GTCCC': 4.9, 'AATCC': 4.19, 'GGCAG': 5.41, 'TACTG': 5.13, 'TACTC': 4.69, 'CCGGG': 5.19, 'AGCGT': 4.86, 'AGCGC': 5.18, 'AGCGA': 5.19, 'AGCGG': 5.15, 'GATTG': 4.81, 'TTTAC': 5.91, 'CAGGG': 4.85, 'TTTAG': 5.82, 'TCAGG': 5.47, 'CTTCG': 4.92, 'TCAGA': 5.47, 'TCAGC': 5.74, 'AAAAT': 3.63, 'TTGGC': 5.56, 'GACAC': 5.49, 'GACAG': 5.26, 'AAAAA': 3.38, 'TTGTG': 5.2, 'AAAAC': 4.05, 'TTGTC': 5.57, 'AAAAG': 3.68, 'GGATG': 4.84, 'GGGGG': 4.75, 'AGTGA': 5.22, 'AGTGC': 5.3, 'AGTGG': 5.31, 'TTTTG': 4.76, 'TTTTC': 4.35, 'AGTGT': 5.1, 'ATGGG': 5.19, 'ATGGC': 5.23, 'ATGGA': 5.34, 'CCAGG': 5.4, 'ATGGT': 4.97, 'GAACC': 4.46, 'GAACG': 4.74, 'AACCA': 4.33, 'AACCC': 4.03, 'AACCG': 4.36, 'ATCGT': 4.8, 'AACCT': 3.64, 'ATCGC': 5.28, 'ATCGA': 5.23, 'ATCGG': 5.25, 'GTAGG': 5.76, 'GTAGC': 5.88, 'GCCTG': 4.94, 'CTAAG': 5.49, 'GTCGC': 5.56, 'GGGTG': 4.77, 'GCACG': 5.74, 'TTATC': 5.89, 'TTATG': 6.02, 'AGAAA': 4.74, 'ACCTC': 4.37, 'ACCTA': 4.68, 'ACCTG': 4.54, 'CAGCG': 5.13, 'ATAAC': 5.62, 'ATAAA': 5.66, 'ATAAG': 5.48, 'TCACG': 5.71, 'CTTGG': 5.31, 'GAGTC': 4.53, 'GAGTG': 4.78, 'TGAAG': 5.15, 'ACTGG': 5.32, 'ACTGA': 5.48, 'ATAAT': 5.28, 'ACTGC': 5.4, 'GCCAG': 5.43, 'TGGAC': 5.19, 'GTGAC': 6.01, 'GTGAG': 5.8, 'AGAAG': 4.5, 'GATCG': 4.96, 'AGAAC': 4.85, 'CGCGG': 5.51, 'GATCC': 4.63, 'TTAAC': 5.85, 'TTAAA': 5.73, 'TTAAG': 5.58, 'TCGAC': 5.64, 'AGCCG': 4.83, 'ATGCC': 5.33, 'ATGCA': 5.55, 'AGCCC': 4.61, 'ATGCG': 5.6, 'AGCCA': 4.53, 'TCGAG': 5.42, 'TTCCC': 4.73, 'CTGAG': 5.56, 'TTCCA': 4.97, 'TTCCG': 5.09, 'CAGTG': 4.98, 'ACCAG': 5.19, 'ACCAA': 5.08, 'ACCAC': 5.38, 'TAGAG': 5.41, 'TAGAC': 5.32, 'TAGAA': 5.33, 'TATCC': 4.98, 'TATCA': 5.38, 'CTCCG': 5.01, 'TATCG': 5.32, 'TGCTC': 5.1, 'ACGTG': 4.85, 'TCCTG': 4.92, 'GTACC': 5.84, 'TCCTC': 4.8, 'GTACG': 5.93, 'TGCTG': 5.27, 'GCGGG': 5.3, 'GCGGC': 5.37, 'GCAGG': 5.52, 'AGCTG': 4.8, 'TCTCG': 5.17, 'ACGTA': 4.84, 'GCAGC': 5.67, 'AGCTC': 4.63, 'TCTCC': 4.92, 'AGCTA': 4.88, 'ATGCT': 5.23, 'GTTGC': 5.7, 'ACACG': 5.47, 'ACACA': 5.5, 'ACACC': 5.31, 'ATATG': 5.32, 'ATATA': 5.76, 'ATATC': 5.4, 'GAGAG': 4.93, 'GAGAC': 5.04, 'GAATG': 4.39, 'GAATC': 4.36, 'ATTTA': 4.75, 'ATTTC': 4.27, 'GGGAG': 4.85, 'ATTTG': 4.12, 'ACTCA': 4.89, 'TGTGG': 5.8, 'ACTCC': 4.73, 'TGTGA': 5.84, 'TGTGC': 5.94, 'ACTCG': 4.93, 'TATGG': 5.88, 'GGCGG': 5.29, 'TATGC': 5.76, 'GGCGC': 5.41, 'CAATG': 4.69, 'TAAAG': 4.7, 'TAAAA': 4.89, 'TAAAC': 5.1, 'CACCG': 4.92, 'GATGC': 5.38, 'GATGG': 5.53, 'AAGCG': 4.63, 'AAGCA': 4.61, 'AAGCC': 4.17, 'GCTAG': 5.67, 'AAGCT': 4.14, 'AACTG': 4.49, 'AACTA': 4.24, 'AACTC': 3.95, 'TTCGG': 5.38, 'GTTCG': 5.1, 'TTCGC': 5.53, 'TTCGA': 5.75, 'CGTGG': 5.64, 'GTTCC': 4.84, 'TGTCC': 5.02, 'TGTCG': 5.28, 'TACGG': 5.33, 'TACGA': 5.61, 'TACGC': 5.32, 'AGTAG': 5.2, 'CAAAG': 4.52, 'AGTAC': 5.54, 'AGTAA': 5.37, 'AGGCC': 4.72, 'AGGCA': 4.82, 'AGGCG': 4.77, 'GCGCG': 5.54, 'CATGG': 5.34, 'GGAAG': 4.63, 'TCTGC': 5.75, 'TCTGG': 5.56, 'AGGCT': 4.33, 'ACAGA': 5.33, 'ACAGC': 5.31, 'ACAGG': 5.29, 'ATTAC': 5.44, 'ATTAA': 5.58, 'ATTAG': 5.57, 'ACAGT': 5.2, 'AATAA': 5.53, 'AATAC': 5.3, 'AATAG': 4.65, 'TCCCG': 4.95, 'ACGGT': 4.73, 'TCCCC': 4.76, 'ACGGC': 5.08, 'ACGGA': 5.06, 'CTCGG': 5.65, 'AATAT': 4.8, 'TTGAG': 5.9, 'GTCAG': 5.67, 'TGATC': 4.77, 'TGATG': 5.22, 'GGCCG': 4.96, 'TTGAA': 6.0, 'TTGAC': 5.91, 'GTTTG': 4.95, 'ACGGG': 4.92, 'TCGGC': 5.43, 'TCGGA': 5.36, 'CGACG': 5.19, 'AAGGA': 4.31, 'AAGGC': 4.51, 'TTTGG': 5.42, 'TTTGA': 5.6, 'AAGGG': 4.42, 'TTTGC': 5.41, 'AAGGT': 3.75, 'AATTC': 3.75, 'AATTA': 4.36, 'AATTG': 4.24, 'AAAGT': 3.35, 'GACGC': 5.09, 'GACGG': 5.1, 'AAAGC': 4.03, 'TCGTC': 5.16, 'AAAGA': 4.02, 'AAAGG': 4.05, 'GGTAG': 5.55, 'ACGCG': 5.2, 'TGCAC': 5.94, 'ACGCC': 5.08, 'ACGCA': 5.27, 'TGCAG': 5.79, 'TAGTA': 5.33, 'TCGTG': 5.29, 'TAGTC': 5.08, 'TAGTG': 5.11, 'GCATG': 5.5, 'TACCA': 5.13, 'TACCC': 4.71, 'TACCG': 4.96, 'AGAGG': 4.97, 'AGGGT': 4.29, 'AGAGA': 5.0, 'AGAGC': 5.04, 'AGGGG': 4.62, 'AGAGT': 4.86, 'AGGGC': 4.73, 'AGGGA': 4.68, 'CATCG': 4.94, 'CACGG': 5.1, 'TTCTC': 4.67, 'TTCTG': 5.14, 'TCCGC': 5.27, 'TCCGG': 5.39, 'ATCAG': 5.34, 'ATCAA': 5.52, 'ATCAC': 5.35, 'AAACG': 4.43, 'CTACG': 5.77, 'AAACC': 4.06, 'ACGTC': 4.7, 'AAACA': 4.65, 'ACATA': 5.14, 'AAACT': 3.85, 'GTTGG': 5.42, 'GCAAG': 5.45, 'CGGGG': 4.94, 'CGAGG': 5.18, 'TTTCA': 5.4, 'TTTCC': 4.63, 'GGTTG': 4.82, 'CAGAG': 5.14, 'TTTCG': 4.98, 'TCAAC': 5.65, 'TCAAG': 5.4, 'ATAGG': 5.6, 'ATAGA': 5.69, 'ATAGC': 5.37, 'ATAGT': 4.86, 'TATTC': 4.51, 'TATTG': 5.02, 'GACCG': 4.69, 'GACCC': 4.45}
	complimentMatrix = {'A':'T', 'T':'A',
				  		'G':'C', 'C':'G'}

	maxGroove = max(grooveMapping.values())
	minGroove = min(grooveMapping.values())

	mGroove = []
	DNAarray = list(sequence.upper())

	for index in range(len(DNAarray)-4):
		pentamer = ''.join(DNAarray[index:index+5])

		if pentamer in grooveMapping.keys():
			mGroove.append(grooveMapping[pentamer])

		else:
			pentamerCompliment = ''
			for bp in reversed(DNAarray[index:index+5]):
				pentamerCompliment = pentamerCompliment + complimentMatrix[bp]
			mGroove.append(grooveMapping[pentamerCompliment])

	# running average of window = 3
	weights = np.repeat(1.0, window)/float(window)
	runningAverage = np.convolve(mGroove, weights, 'valid')

	# normalizing data
	normalize = (ymax - ymin)*(runningAverage - minGroove)/(maxGroove - minGroove) + ymin;

	# pulling out feature indecies
	middle = len(runningAverage)/2 + len(runningAverage)%2 - 1
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	return mGroove, runningAverage[start-overhang:end+overhang], normalize[start-overhang:end+overhang]


def helixTwist(sequence, recLength = 5, overhang = 12, window = 3,
			   ymax = 1, ymin = -1):
	"""
	Takes a string of DNA in and converts it to its bp helical twist
	"""
	hTwistMapping = {'AAATG': [36.86, 33.42], 'GCCCG': [33.78, 33.44], 'AAATC': [36.55, 33.42], 'AAATA': [37.22, 33.85], 'AAATT': [37.52, 35.11], 'GCGTG': [33.34, 33.73], 'TGGTG': [34.81, 34.27], 'GGCTG': [37.17, 31.16], 'TGGTC': [34.3, 34.68], 'GTGGC': [34.2, 33.11], 'AGACA': [36.49, 34.36], 'GATAG': [32.43, 33.89], 'AGACC': [36.1, 34.74], 'GATAC': [32.1, 34.46], 'AGACG': [36.41, 34.44], 'AAGTT': [33.58, 36.3], 'TTACG': [34.26, 33.58], 'TTACA': [34.6, 33.74], 'AGACT': [36.01, 34.89], 'TTACC': [34.64, 34.56], 'AGCAA': [36.94, 34.41], 'TCGCC': [32.6, 36.2], 'AGCAC': [36.67, 34.03], 'TCGCG': [32.65, 35.66], 'AGCAG': [36.97, 33.86], 'TTCAG': [35.08, 34.21], 'TTCAC': [34.8, 34.78], 'TCGGG': [32.86, 33.58], 'GAAAG': [35.91, 36.1], 'TCATG': [35.0, 31.75], 'GAAAC': [36.01, 36.17], 'TCATC': [34.64, 32.03], 'AACAG': [34.88, 34.91], 'AACAC': [34.68, 35.07], 'AACAA': [34.6, 35.64], 'TGTTG': [34.7, 35.19], 'TGTTC': [34.81, 35.68], 'GACTG': [34.65, 31.46], 'TATAG': [31.96, 34.63], 'AACAT': [35.12, 35.94], 'TATAA': [31.78, 34.57], 'TATAC': [31.78, 34.63], 'GTCGG': [35.42, 32.81], 'CTAGG': [34.19, 31.47], 'TAATG': [35.43, 32.4], 'TGGCG': [33.36, 36.34], 'TAATC': [35.28, 32.72], 'TGGCC': [33.23, 37.04], 'TAATA': [35.37, 32.48], 'TGGCA': [33.34, 36.44], 'GCTTG': [32.11, 34.57], 'AATGT': [33.25, 36.13], 'CGGCG': [33.59, 36.24], 'AATGG': [32.12, 34.87], 'GTGGG': [34.09, 33.55], 'AATGC': [32.67, 35.25], 'AATGA': [32.87, 34.98], 'ATACA': [34.63, 34.26], 'ATACC': [34.56, 34.69], 'ATACG': [34.59, 34.16], 'ACTAC': [31.68, 34.24], 'GGGCG': [33.85, 36.18], 'ACTAA': [31.57, 34.45], 'ACTAG': [32.22, 35.0], 'GGGCC': [33.57, 36.61], 'TGACG': [35.16, 34.29], 'ATACT': [34.44, 34.73], 'TGACA': [35.3, 34.2], 'TGACC': [35.47, 34.83], 'GCCGG': [33.41, 32.61], 'AGTTA': [35.6, 34.85], 'AGTTC': [35.93, 35.24], 'AGTTG': [36.24, 34.76], 'AGGTC': [35.04, 34.84], 'AGGTA': [35.24, 34.69], 'AGGTG': [35.25, 34.31], 'TAACG': [35.27, 34.89], 'TAACC': [35.08, 35.3], 'TAACA': [35.32, 34.78], 'GTGCC': [34.22, 36.31], 'GTGCG': [34.69, 35.54], 'CACAG': [33.74, 35.2], 'AACGC': [35.06, 33.41], 'GTGTG': [35.12, 33.35], 'GCTCG': [31.41, 35.31], 'ACATC': [35.37, 32.37], 'ATGAT': [34.82, 36.14], 'ACATG': [35.7, 32.0], 'TCTTC': [32.42, 35.03], 'TCTTG': [32.47, 34.77], 'ATGAA': [35.03, 35.39], 'ATGAC': [35.1, 35.47], 'TATGA': [31.46, 35.1], 'ATGAG': [34.93, 35.1], 'ACCCC': [34.28, 33.44], 'ACCCA': [34.63, 33.53], 'CTGCG': [34.24, 35.64], 'ACCCG': [34.56, 33.51], 'GGTGG': [33.92, 34.27], 'GGTGC': [34.16, 34.26], 'ACTTG': [33.13, 34.94], 'TAGCG': [31.38, 36.72], 'ACTTA': [33.06, 35.16], 'TAGCA': [31.39, 36.48], 'ACTTC': [32.77, 35.22], 'TAGCC': [31.35, 37.31], 'TGAGA': [35.17, 31.77], 'TGAGC': [34.81, 31.45], 'TGAGG': [34.61, 31.54], 'ATCTG': [36.37, 31.12], 'GTAAC': [34.23, 35.06], 'ATCTC': [36.47, 31.47], 'ATCTA': [36.79, 31.29], 'GTAAG': [34.34, 34.87], 'CAACG': [35.01, 35.1], 'GGAGC': [35.75, 31.38], 'AGGAG': [34.13, 36.23], 'GGAGG': [35.57, 31.42], 'AGGAC': [34.49, 36.03], 'TCTAC': [30.99, 34.42], 'TGGGC': [33.45, 33.75], 'TGGGA': [33.23, 33.69], 'TCTAG': [31.57, 34.35], 'TGGGG': [33.26, 33.49], 'ATGTC': [35.42, 34.24], 'ATGTA': [35.76, 34.06], 'ATGTG': [35.86, 33.53], 'ACAAC': [35.22, 35.25], 'ATTCT': [36.11, 36.6], 'ACAAA': [35.41, 35.77], 'AAGTC': [33.16, 35.44], 'ACAAG': [35.43, 35.05], 'TTAGG': [34.35, 31.45], 'AAGTG': [33.1, 34.91], 'TCCAC': [33.75, 34.34], 'AATCT': [33.3, 37.53], 'GAGCC': [31.39, 37.09], 'TTGCG': [34.85, 35.42], 'ATTCG': [35.86, 35.52], 'ATTCA': [36.06, 35.03], 'ATTCC': [35.6, 35.99], 'TGCCC': [36.27, 33.82], 'AGGAA': [34.39, 36.26], 'AATCA': [33.01, 36.14], 'AATCG': [32.89, 36.19], 'ATTGC': [35.34, 34.49], 'TTGGA': [34.57, 33.66], 'TTGGG': [34.28, 32.76], 'TGCGG': [35.63, 33.13], 'TGCGC': [35.52, 32.79], 'TGCGA': [35.53, 33.31], 'GGGGC': [33.56, 33.68], 'GTCTG': [36.44, 31.42], 'TAGGA': [31.66, 34.64], 'TAGGC': [31.51, 34.04], 'TAGGG': [31.16, 34.12], 'AGATG': [36.48, 32.25], 'AGATA': [36.52, 32.48], 'AGATC': [36.68, 32.82], 'TAAGC': [34.85, 32.17], 'TAAGA': [34.89, 32.43], 'TAAGG': [34.66, 32.34], 'TGGAG': [33.85, 35.8], 'AAGAC': [32.56, 36.4], 'AAGAA': [32.3, 36.52], 'AAGAG': [32.31, 35.92], 'GCTGG': [30.94, 34.09], 'TCACC': [34.15, 34.04], 'AAGAT': [32.88, 37.56], 'CTGGG': [33.88, 33.54], 'ACCGG': [34.12, 32.44], 'ACCGC': [34.31, 32.31], 'GGTCG': [34.74, 35.73], 'ACCGA': [34.18, 32.64], 'TTAGA': [34.4, 31.27], 'AAGTA': [33.24, 35.21], 'TGTAG': [33.91, 35.06], 'TGTAC': [33.68, 34.52], 'TTAGC': [34.68, 31.43], 'TACAC': [33.6, 34.99], 'TACAA': [33.4, 35.27], 'TACAG': [33.82, 34.82], 'TTGCA': [34.61, 35.49], 'AGTCG': [35.27, 35.38], 'AGTCA': [35.06, 35.14], 'CAAGG': [34.98, 32.51], 'AGTCC': [34.97, 35.78], 'GCGAG': [32.59, 35.13], 'CATAG': [31.84, 35.03], 'GGACG': [36.13, 34.01], 'GGACC': [35.8, 34.71], 'TGAAC': [35.12, 35.57], 'AACGT': [35.24, 34.53], 'ATTGT': [35.45, 35.55], 'GAAGG': [34.75, 32.02], 'GTTAG': [35.18, 34.54], 'GAAGC': [34.96, 32.06], 'ATTGA': [34.93, 34.6], 'AACGG': [35.14, 33.45], 'AACGA': [34.71, 33.66], 'ATTGG': [34.93, 34.4], 'GAGGG': [31.33, 33.87], 'ATCCT': [36.16, 34.37], 'GAGGC': [31.38, 33.92], 'ATCCG': [36.59, 32.93], 'TCCAG': [33.74, 33.98], 'GTATG': [34.49, 31.46], 'ATCCC': [36.09, 33.66], 'ATCCA': [36.24, 33.69], 'ACGAA': [33.73, 36.0], 'GAGCG': [31.47, 36.68], 'ACGAC': [33.73, 35.7], 'TTGCC': [34.5, 36.34], 'TGCCG': [36.2, 33.58], 'ACGAG': [33.69, 35.94], 'GTCCG': [36.17, 33.8], 'GTCCC': [35.97, 33.99], 'AATCC': [33.31, 36.57], 'GGCAG': [36.22, 33.75], 'TACTG': [34.47, 31.52], 'TACTC': [34.56, 31.76], 'CCGGG': [32.53, 33.24], 'AGCGT': [37.02, 33.0], 'AGCGC': [36.76, 32.08], 'AGCGA': [36.83, 32.44], 'AGCGG': [36.72, 32.34], 'GATTG': [32.75, 34.92], 'TTTAC': [35.53, 34.25], 'CAGGG': [31.11, 34.07], 'TTTAG': [35.72, 34.35], 'TCAGG': [34.07, 30.94], 'CTTCG': [35.11, 35.23], 'TCAGA': [34.38, 31.28], 'TCAGC': [34.29, 31.14], 'AAAAT': [36.93, 37.68], 'TTGGC': [34.72, 33.0], 'GACAC': [34.03, 34.92], 'GACAG': [34.25, 34.79], 'AAAAA': [37.74, 38.01], 'TTGTG': [35.72, 33.59], 'AAAAC': [37.13, 36.95], 'TTGTC': [34.98, 33.76], 'AAAAG': [37.02, 37.18], 'GGATG': [36.16, 32.03], 'GGGGG': [33.25, 33.41], 'AGTGA': [34.23, 33.87], 'AGTGC': [34.58, 33.99], 'AGTGG': [34.38, 34.11], 'TTTTG': [36.92, 35.51], 'TTTTC': [37.0, 36.32], 'AGTGT': [34.67, 34.61], 'ATGGG': [34.56, 33.45], 'ATGGC': [34.93, 33.39], 'ATGGA': [34.46, 33.69], 'CCAGG': [34.05, 31.17], 'ATGGT': [34.88, 34.55], 'GAACC': [35.45, 35.19], 'GAACG': [35.71, 34.9], 'AACCA': [35.26, 34.89], 'AACCC': [35.8, 34.89], 'AACCG': [35.23, 34.48], 'ATCGT': [36.22, 33.64], 'AACCT': [35.92, 35.61], 'ATCGC': [35.82, 32.49], 'ATCGA': [35.76, 32.77], 'ATCGG': [35.6, 32.57], 'GTAGG': [34.06, 31.41], 'GTAGC': [34.48, 31.25], 'GCCTG': [33.99, 31.14], 'CTAAG': [34.71, 35.04], 'GTCGC': [35.46, 32.7], 'GGGTG': [34.41, 34.14], 'GCACG': [34.75, 33.78], 'TTATC': [34.59, 31.94], 'TTATG': [34.54, 31.48], 'AGAAA': [36.43, 36.02], 'ACCTC': [34.93, 31.49], 'ACCTA': [34.99, 31.46], 'ACCTG': [35.08, 31.38], 'CAGCG': [31.2, 36.67], 'ATAAC': [34.54, 35.5], 'ATAAA': [34.6, 35.94], 'ATAAG': [35.15, 35.03], 'TCACG': [34.66, 33.8], 'CTTGG': [34.56, 34.61], 'GAGTC': [31.83, 34.87], 'GAGTG': [31.79, 34.12], 'TGAAG': [35.1, 35.14], 'ACTGG': [31.51, 34.39], 'ACTGA': [31.46, 34.41], 'ATAAT': [35.02, 35.9], 'ACTGC': [31.49, 33.95], 'GCCAG': [33.53, 34.28], 'TGGAC': [33.79, 36.11], 'GTGAC': [34.6, 35.33], 'GTGAG': [34.48, 34.92], 'AGAAG': [36.79, 35.47], 'GATCG': [32.29, 35.54], 'AGAAC': [36.45, 35.7], 'CGCGG': [35.54, 32.35], 'GATCC': [32.41, 36.29], 'TTAAC': [34.36, 35.05], 'TTAAA': [34.67, 35.63], 'TTAAG': [34.81, 34.68], 'TCGAC': [33.06, 35.51], 'AGCCG': [37.23, 33.27], 'ATGCC': [34.66, 36.28], 'ATGCA': [35.12, 35.6], 'AGCCC': [37.3, 33.51], 'ATGCG': [35.0, 35.58], 'AGCCA': [37.62, 33.35], 'TCGAG': [33.01, 35.17], 'TTCCC': [36.01, 33.66], 'CTGAG': [34.47, 34.84], 'TTCCA': [36.04, 33.8], 'TTCCG': [35.82, 34.24], 'CAGTG': [31.61, 34.3], 'ACCAG': [34.39, 34.21], 'ACCAA': [34.7, 34.6], 'ACCAC': [34.37, 34.18], 'TAGAG': [31.4, 35.31], 'TAGAC': [31.64, 36.16], 'TAGAA': [31.57, 36.55], 'TATCC': [32.31, 36.04], 'TATCA': [31.94, 35.75], 'CTCCG': [35.79, 33.51], 'TATCG': [31.89, 35.68], 'TGCTC': [36.61, 31.46], 'ACGTG': [34.48, 34.05], 'TCCTG': [34.22, 31.21], 'GTACC': [34.23, 34.34], 'TCCTC': [34.17, 31.53], 'GTACG': [34.2, 33.97], 'TGCTG': [36.65, 31.23], 'GCGGG': [32.3, 33.41], 'GCGGC': [32.46, 33.44], 'GCAGG': [33.94, 31.18], 'AGCTG': [37.48, 31.01], 'TCTCG': [31.51, 35.18], 'ACGTA': [34.34, 34.3], 'GCAGC': [34.49, 31.29], 'AGCTC': [37.66, 31.33], 'TCTCC': [31.48, 35.79], 'AGCTA': [37.56, 31.13], 'ATGCT': [34.56, 36.89], 'GTTGC': [35.02, 34.8], 'ACACG': [35.35, 33.84], 'ACACA': [35.28, 33.88], 'ACACC': [34.83, 34.36], 'ATATG': [35.37, 32.04], 'ATATA': [34.46, 32.12], 'ATATC': [35.01, 32.32], 'GAGAG': [31.39, 35.65], 'GAGAC': [31.76, 36.17], 'GAATG': [35.63, 32.69], 'GAATC': [35.53, 32.93], 'ATTTA': [36.55, 35.79], 'ATTTC': [36.7, 35.77], 'GGGAG': [33.49, 35.58], 'ATTTG': [37.28, 35.49], 'ACTCA': [32.14, 35.29], 'TGTGG': [33.07, 34.79], 'ACTCC': [31.81, 35.76], 'TGTGA': [33.59, 34.48], 'TGTGC': [33.55, 35.01], 'ACTCG': [31.96, 35.21], 'TATGG': [31.26, 34.97], 'GGCGG': [36.31, 32.31], 'TATGC': [31.62, 34.78], 'GGCGC': [36.24, 32.32], 'CAATG': [35.08, 32.41], 'TAAAG': [35.7, 36.11], 'TAAAA': [35.6, 36.72], 'TAAAC': [35.72, 36.4], 'CACCG': [34.05, 34.36], 'GATGC': [31.96, 34.59], 'GATGG': [31.66, 34.54], 'AAGCG': [32.27, 37.15], 'AAGCA': [32.29, 37.13], 'AAGCC': [32.49, 37.74], 'GCTAG': [31.36, 34.66], 'AAGCT': [32.14, 38.05], 'AACTG': [35.45, 31.73], 'AACTA': [35.56, 32.1], 'AACTC': [36.03, 32.41], 'TTCGG': [35.61, 32.67], 'GTTCG': [35.58, 35.61], 'TTCGC': [35.53, 32.71], 'TTCGA': [34.95, 32.98], 'CGTGG': [33.53, 34.18], 'GTTCC': [35.48, 36.02], 'TGTCC': [34.09, 36.22], 'TGTCG': [34.23, 35.73], 'TACGG': [33.73, 33.27], 'TACGA': [33.57, 33.73], 'TACGC': [33.93, 33.14], 'AGTAG': [35.18, 34.13], 'CAAAG': [35.41, 36.3], 'AGTAC': [34.74, 33.82], 'AGTAA': [34.75, 34.55], 'AGGCC': [33.71, 36.64], 'AGGCA': [34.03, 36.26], 'AGGCG': [34.34, 36.31], 'GCGCG': [32.44, 35.7], 'CATGG': [31.61, 34.66], 'GGAAG': [35.39, 35.02], 'TCTGC': [31.17, 33.98], 'TCTGG': [31.02, 33.91], 'AGGCT': [34.0, 37.47], 'ACAGA': [35.0, 31.6], 'ACAGC': [35.03, 31.35], 'ACAGG': [34.61, 31.34], 'ATTAC': [35.53, 34.68], 'ATTAA': [35.37, 34.44], 'ATTAG': [35.33, 34.46], 'ACAGT': [34.72, 31.75], 'AATAA': [32.44, 34.83], 'AATAC': [32.78, 34.79], 'AATAG': [33.54, 35.89], 'TCCCG': [34.12, 33.56], 'ACGGT': [33.29, 34.47], 'TCCCC': [33.65, 33.55], 'ACGGC': [33.55, 33.79], 'ACGGA': [33.01, 34.6], 'CTCGG': [34.73, 32.31], 'AATAT': [33.17, 35.25], 'TTGAG': [34.65, 34.56], 'GTCAG': [35.2, 34.37], 'TGATC': [35.6, 32.48], 'TGATG': [35.36, 31.93], 'GGCCG': [36.85, 33.23], 'TTGAA': [34.59, 34.8], 'TTGAC': [34.61, 34.78], 'GTTTG': [36.33, 35.28], 'ACGGG': [33.23, 33.75], 'TCGGC': [32.47, 32.95], 'TCGGA': [32.77, 33.64], 'CGACG': [35.57, 34.26], 'AAGGA': [32.55, 34.9], 'AAGGC': [32.34, 34.57], 'TTTGG': [35.2, 34.57], 'TTTGA': [35.3, 34.64], 'AAGGG': [31.94, 34.22], 'TTTGC': [35.29, 34.35], 'AAGGT': [32.53, 35.68], 'AATTC': [33.75, 36.17], 'AATTA': [33.31, 35.92], 'AATTG': [33.37, 35.62], 'AAAGT': [36.93, 33.88], 'GACGC': [34.21, 33.02], 'GACGG': [34.1, 33.08], 'AAAGC': [36.56, 32.81], 'TCGTC': [33.65, 34.18], 'AAAGA': [36.69, 33.18], 'AAAGG': [35.88, 32.79], 'GGTAG': [34.52, 34.3], 'ACGCG': [33.2, 35.9], 'TGCAC': [35.46, 34.64], 'ACGCC': [33.32, 36.67], 'ACGCA': [33.71, 35.83], 'TGCAG': [35.64, 34.5], 'TAGTA': [31.63, 34.42], 'TCGTG': [33.99, 33.77], 'TAGTC': [31.53, 34.57], 'TAGTG': [31.7, 34.28], 'GCATG': [34.82, 31.54], 'TACCA': [34.38, 34.3], 'TACCC': [34.53, 34.33], 'TACCG': [34.43, 34.27], 'AGAGG': [35.39, 31.44], 'AGGGT': [34.17, 34.42], 'AGAGA': [35.8, 31.7], 'AGAGC': [35.84, 31.62], 'AGGGG': [33.78, 33.61], 'AGAGT': [35.66, 31.76], 'AGGGC': [34.0, 33.54], 'AGGGA': [33.86, 33.6], 'CATCG': [32.14, 35.83], 'CACGG': [33.81, 33.7], 'TTCTC': [36.61, 31.51], 'TTCTG': [36.46, 31.59], 'TCCGC': [33.51, 32.42], 'TCCGG': [33.6, 32.84], 'ATCAG': [35.8, 34.24], 'ATCAA': [35.38, 34.6], 'ATCAC': [35.64, 34.03], 'AAACG': [36.57, 35.48], 'CTACG': [34.25, 34.02], 'AAACC': [36.34, 36.2], 'ACGTC': [34.3, 34.58], 'AAACA': [36.59, 35.22], 'ACATA': [35.48, 32.21], 'AAACT': [35.86, 36.86], 'GTTGG': [31.76, 34.44], 'GCAAG': [34.37, 34.89], 'CGGGG': [33.22, 33.62], 'CGAGG': [34.97, 31.5], 'TTTCA': [35.9, 34.89], 'TTTCC': [35.97, 36.46], 'GGTTG': [35.52, 31.89], 'CAGAG': [31.18, 35.6], 'TTTCG': [35.88, 35.62], 'TCAAC': [34.69, 34.93], 'TCAAG': [34.53, 34.76], 'ATAGG': [34.5, 31.71], 'ATAGA': [34.25, 31.88], 'ATAGC': [34.96, 31.21], 'ATAGT': [35.53, 31.86], 'TATTC': [32.92, 35.93], 'TATTG': [32.4, 35.12], 'GACCG': [34.65, 34.17], 'GACCC': [34.81, 34.36]}
	complimentMatrix = {'A':'T', 'T':'A',
				  		'G':'C', 'C':'G'}

	hTwists = np.array(hTwistMapping.values()).flatten()
	maxTwist = max(hTwists)
	minTwist = min(hTwists)

	hTwist = []
	DNAarray = list(sequence.upper())
	previousTwists = []
	compliment = False
	for index in range(len(DNAarray)-4):
		pentamer = ''.join(DNAarray[index:index+5])

		if pentamer not in hTwistMapping.keys():
			compliment = True
			pentamerCompliment = ''

			for bp in reversed(DNAarray[index:index+5]):
				pentamerCompliment = pentamerCompliment + complimentMatrix[bp]

			pentamer = pentamerCompliment

		twists = copy.copy(hTwistMapping[pentamer])
		if compliment:
			twists.reverse()

		if index == 0:
			hTwist.append(twists[0])
		else:
			hTwist.append(np.mean([twists[0],
								   previousTwists[1]]))
		if index == len(DNAarray)-5:
			hTwist.append(twists[1])

		previousTwists = twists
		compliment = False

	# running average of window = 3
	weights = np.repeat(1.0, window)/float(window)
	runningAverage = np.convolve(hTwist, weights, 'valid')

	# normalizing data
	normalize = (ymax - ymin)*(runningAverage - minTwist)/(maxTwist - minTwist) + ymin;

	# helix calculation adds one digit, original length list = 11, new list = 8
	middle = len(runningAverage)/2 + len(runningAverage)%2
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	return hTwist, runningAverage[start-overhang:end+overhang], normalize[start-overhang:end+overhang]


def roll(sequence, recLength = 5, overhang = 12, window = 3,
		 ymax = 1, ymin = -1):
	"""
	Takes a string of DNA in and converts it to its bp roll
	"""
	rollMapping = {'AAATG': [-4.21, -7.0], 'GCCCG': [-1.71, -1.7], 'AAATC': [-2.43, -5.36], 'AAATA': [-3.91, -6.42], 'AAATT': [-5.0, -8.57], 'GCGTG': [4.71, -2.23], 'TGGTG': [-0.68, -2.83], 'GGCTG': [-2.29, -2.64], 'TGGTC': [-0.69, -2.73], 'GTGGC': [3.09, -1.52], 'AGACA': [-1.88, -2.8], 'GATAG': [-4.19, 6.28], 'AGACC': [-1.9, -3.14], 'GATAC': [-3.62, 5.72], 'AGACG': [-2.02, -2.95], 'AAGTT': [-3.62, -5.19], 'TTACG': [6.35, -1.95], 'TTACA': [6.03, -2.1], 'AGACT': [-2.0, -3.59], 'TTACC': [5.01, -3.71], 'AGCAA': [-3.02, 3.78], 'TCGCC': [3.27, -1.65], 'AGCAC': [-1.72, 2.81], 'TCGCG': [4.5, -1.33], 'AGCAG': [-2.72, 3.76], 'TTCAG': [0.57, 5.37], 'TTCAC': [0.69, 4.07], 'TCGGG': [3.35, -1.75], 'GAAAG': [-3.0, -4.95], 'TCATG': [5.91, -4.44], 'GAAAC': [-2.18, -3.35], 'TCATC': [4.02, -4.68], 'AACAG': [-2.4, 5.23], 'AACAC': [-2.42, 4.19], 'AACAA': [-2.29, 6.26], 'TGTTG': [-2.5, -2.78], 'TGTTC': [-2.74, -3.14], 'GACTG': [-2.08, -1.4], 'TATAG': [-3.1, 5.66], 'AACAT': [-3.23, 5.76], 'TATAA': [-3.01, 7.31], 'TATAC': [-2.8, 5.6], 'GTCGG': [-1.52, 3.41], 'CTAGG': [4.92, -2.23], 'TAATG': [-2.15, -4.86], 'TGGCG': [-1.66, -1.9], 'TAATC': [-2.32, -4.51], 'TGGCC': [-1.76, -2.42], 'TAATA': [-2.19, -3.92], 'TGGCA': [-1.33, -1.81], 'GCTTG': [-2.39, -3.44], 'AATGT': [-6.17, 4.78], 'CGGCG': [-1.33, -1.46], 'AATGG': [-4.16, 3.6], 'GTGGG': [2.54, -1.85], 'AATGC': [-5.22, 3.59], 'AATGA': [-6.08, 3.8], 'ATACA': [5.89, -2.07], 'ATACC': [5.39, -2.59], 'ATACG': [5.89, -2.22], 'ACTAC': [-1.63, 4.89], 'GGGCG': [-1.63, -1.66], 'ACTAA': [-0.98, 7.88], 'ACTAG': [-2.86, 3.55], 'GGGCC': [-2.36, -2.4], 'TGACG': [-1.16, -2.4], 'ATACT': [6.22, -2.71], 'TGACA': [-1.53, -2.49], 'TGACC': [-2.01, -3.46], 'GCCGG': [-1.33, 2.08], 'AGTTA': [-3.98, -3.3], 'AGTTC': [-4.54, -3.79], 'AGTTG': [-4.47, -3.88], 'AGGTC': [-2.46, -3.96], 'AGGTA': [-2.81, -4.49], 'AGGTG': [-1.65, -3.26], 'TAACG': [-2.63, -2.83], 'TAACC': [-3.03, -3.26], 'TAACA': [-2.39, -2.47], 'GTGCC': [3.05, -1.35], 'GTGCG': [3.27, -0.64], 'CACAG': [-2.15, 4.99], 'AACGC': [-2.93, 4.64], 'GTGTG': [5.01, -1.37], 'GCTCG': [-2.22, -1.22], 'ACATC': [5.15, -4.91], 'ATGAT': [3.79, -1.32], 'ACATG': [5.73, -4.67], 'TCTTC': [-2.28, -3.52], 'TCTTG': [-1.59, -3.37], 'ATGAA': [5.73, 0.04], 'ATGAC': [5.16, -2.06], 'TATGA': [-3.24, 4.95], 'ATGAG': [5.69, -1.52], 'ACCCC': [-1.8, -2.4], 'ACCCA': [-1.42, -2.0], 'CTGCG': [3.64, -0.89], 'ACCCG': [-0.94, -1.76], 'GGTGG': [-2.55, 2.46], 'GGTGC': [-2.69, 2.97], 'ACTTG': [-2.35, -3.59], 'TAGCG': [-2.2, -1.71], 'ACTTA': [-2.87, -3.58], 'TAGCA': [-2.09, -1.5], 'ACTTC': [-3.34, -4.2], 'TAGCC': [-2.77, -2.39], 'TGAGA': [-1.29, -2.02], 'TGAGC': [-0.9, -2.11], 'TGAGG': [-0.98, -2.49], 'ATCTG': [-0.33, -2.27], 'GTAAC': [5.89, -2.71], 'ATCTC': [-1.46, -2.16], 'ATCTA': [-1.24, -2.21], 'GTAAG': [5.9, -3.24], 'CAACG': [-2.97, -2.83], 'GGAGC': [-1.59, -2.75], 'AGGAG': [-1.32, -2.62], 'GGAGG': [-1.91, -3.27], 'AGGAC': [-0.96, -2.19], 'TCTAC': [-2.44, 4.12], 'TGGGC': [-2.07, -1.99], 'TGGGA': [-1.55, -0.93], 'TCTAG': [-1.22, 5.27], 'TGGGG': [-1.74, -2.1], 'ATGTC': [4.9, -2.83], 'ATGTA': [5.57, -2.14], 'ATGTG': [6.57, -2.22], 'ACAAC': [6.2, -3.26], 'ATTCT': [-3.86, -1.42], 'ACAAA': [6.09, -2.56], 'AAGTC': [-4.01, -5.04], 'ACAAG': [5.65, -3.95], 'TTAGG': [5.87, -2.83], 'AAGTG': [-2.79, -4.51], 'TCCAC': [-0.49, 2.75], 'AATCT': [-6.11, -2.71], 'GAGCC': [-2.76, -3.07], 'TTGCG': [4.82, -0.77], 'ATTCG': [-2.98, -0.24], 'ATTCA': [-2.45, 0.21], 'ATTCC': [-3.73, -0.88], 'TGCCC': [-1.81, -1.78], 'AGGAA': [-1.44, -1.3], 'AATCA': [-4.75, -1.08], 'AATCG': [-4.9, -0.8], 'ATTGC': [-2.92, 3.9], 'TTGGA': [2.7, -1.03], 'TTGGG': [6.64, -0.64], 'TGCGG': [-0.88, 2.75], 'TGCGC': [-0.67, 3.54], 'TGCGA': [-0.91, 4.16], 'GGGGC': [-2.02, -1.96], 'GTCTG': [-1.52, -1.18], 'TAGGA': [-2.51, -0.97], 'TAGGC': [-2.47, -1.96], 'TAGGG': [-2.67, -2.27], 'AGATG': [-1.27, -5.19], 'AGATA': [-1.03, -4.23], 'AGATC': [-1.69, -5.53], 'TAAGC': [-2.87, -2.56], 'TAAGA': [-3.37, -2.51], 'TAAGG': [-3.71, -4.02], 'TGGAG': [-0.86, -1.81], 'AAGAC': [-2.15, -2.37], 'AAGAA': [-1.79, -0.63], 'AAGAG': [-1.76, -1.36], 'GCTGG': [-2.45, 2.75], 'TCACC': [3.82, -3.21], 'AAGAT': [-3.96, -3.29], 'CTGGG': [2.42, -1.97], 'ACCGG': [-0.44, 2.54], 'ACCGC': [-0.62, 2.79], 'GGTCG': [-3.5, -2.17], 'ACCGA': [-0.94, 3.35], 'TTAGA': [6.26, -1.11], 'AAGTA': [-3.52, -4.72], 'TGTAG': [-3.23, 3.24], 'TGTAC': [-1.75, 4.26], 'TTAGC': [5.63, -2.69], 'TACAC': [-2.01, 3.4], 'TACAA': [-1.49, 5.8], 'TACAG': [-1.95, 3.96], 'TTGCA': [4.6, -1.25], 'AGTCG': [-4.46, -2.31], 'AGTCA': [-3.69, -1.77], 'CAAGG': [-3.66, -2.93], 'AGTCC': [-4.26, -2.46], 'GCGAG': [4.38, -1.17], 'CATAG': [-3.65, 5.41], 'GGACG': [-1.8, -2.49], 'GGACC': [-2.12, -3.32], 'TGAAC': [0.09, -2.98], 'AACGT': [-3.43, 5.64], 'ATTGT': [-2.43, 6.33], 'GAAGG': [-3.82, -3.57], 'GTTAG': [-2.89, 5.73], 'GAAGC': [-3.36, -2.79], 'ATTGA': [-2.1, 7.05], 'AACGG': [-3.07, 3.74], 'AACGA': [-2.49, 5.41], 'ATTGG': [-2.07, 5.06], 'GAGGG': [-2.73, -2.62], 'ATCCT': [-1.25, -1.41], 'GAGGC': [-2.42, -2.18], 'ATCCG': [-2.07, -2.34], 'TCCAG': [-0.76, 2.8], 'GTATG': [6.01, -2.99], 'ATCCC': [-1.49, -1.06], 'ATCCA': [-0.96, -0.54], 'ACGAA': [5.06, -0.84], 'GAGCG': [-1.89, -1.98], 'ACGAC': [5.01, -2.42], 'TTGCC': [3.97, -2.26], 'TGCCG': [-1.38, -1.26], 'ACGAG': [4.47, -2.01], 'GTCCG': [-1.73, -0.51], 'GTCCC': [-2.34, -0.93], 'AATCC': [-5.81, -0.9], 'GGCAG': [-1.34, 3.38], 'TACTG': [-2.15, -1.74], 'TACTC': [-3.67, -2.76], 'CCGGG': [1.93, -1.59], 'AGCGT': [-2.63, 3.7], 'AGCGC': [-1.92, 2.89], 'AGCGA': [-2.29, 4.15], 'AGCGG': [-1.9, 2.44], 'GATTG': [-4.28, -2.12], 'TTTAC': [-0.99, 6.47], 'CAGGG': [-3.39, -2.88], 'TTTAG': [-1.41, 6.08], 'TCAGG': [3.63, -3.12], 'CTTCG': [-3.7, -0.16], 'TCAGA': [4.43, -1.76], 'TCAGC': [4.93, -2.0], 'AAAAT': [-3.56, -5.12], 'TTGGC': [4.45, -1.23], 'GACAC': [-2.01, 4.23], 'GACAG': [-2.46, 3.98], 'AAAAA': [-5.05, -5.09], 'TTGTG': [6.07, -2.4], 'AAAAC': [-3.62, -4.8], 'TTGTC': [6.28, -1.95], 'AAAAG': [-4.23, -6.47], 'GGATG': [-1.14, -4.71], 'GGGGG': [-2.24, -2.41], 'AGTGA': [-3.58, 3.56], 'AGTGC': [-3.54, 2.74], 'AGTGG': [-2.95, 2.64], 'TTTTG': [-2.76, -2.19], 'TTTTC': [-4.32, -3.67], 'AGTGT': [-3.51, 3.69], 'ATGGG': [3.38, -2.51], 'ATGGC': [3.68, -2.18], 'ATGGA': [4.34, -0.26], 'CCAGG': [2.39, -2.87], 'ATGGT': [4.04, -1.22], 'GAACC': [-3.83, -3.59], 'GAACG': [-2.93, -2.93], 'AACCA': [-3.3, -0.59], 'AACCC': [-4.18, -1.38], 'AACCG': [-3.13, -0.65], 'ATCGT': [-1.68, 4.65], 'AACCT': [-4.78, -2.23], 'ATCGC': [-0.66, 4.15], 'ATCGA': [-0.54, 5.51], 'ATCGG': [-0.66, 3.63], 'GTAGG': [4.46, -2.71], 'GTAGC': [4.45, -2.28], 'GCCTG': [-1.61, -2.26], 'CTAAG': [6.02, -3.25], 'GTCGC': [-1.39, 4.64], 'GGGTG': [-1.08, -2.99], 'GCACG': [3.22, -2.27], 'TTATC': [7.32, -3.54], 'TTATG': [7.92, -3.28], 'AGAAA': [-0.69, -2.61], 'ACCTC': [-2.35, -3.46], 'ACCTA': [-1.78, -2.96], 'ACCTG': [-1.63, -2.97], 'CAGCG': [-1.93, -1.69], 'ATAAC': [6.14, -3.5], 'ATAAA': [8.13, -0.95], 'ATAAG': [6.31, -3.47], 'TCACG': [4.27, -2.49], 'CTTGG': [-3.4, 4.48], 'GAGTC': [-2.44, -3.57], 'GAGTG': [-1.98, -3.21], 'TGAAG': [0.22, -3.01], 'ACTGG': [-1.82, 3.06], 'ACTGA': [-1.92, 4.75], 'ATAAT': [6.41, -2.76], 'ACTGC': [-1.42, 3.67], 'GCCAG': [-1.52, 2.79], 'TGGAC': [-0.37, -2.08], 'GTGAC': [4.03, -1.42], 'GTGAG': [4.33, -0.77], 'AGAAG': [-1.65, -4.1], 'GATCG': [-4.15, -0.57], 'AGAAC': [-0.56, -3.03], 'CGCGG': [-1.05, 2.68], 'GATCC': [-4.63, -1.67], 'TTAAC': [7.53, -2.86], 'TTAAA': [5.72, -3.0], 'TTAAG': [6.23, -3.71], 'TCGAC': [5.38, -1.63], 'AGCCG': [-2.14, -1.44], 'ATGCC': [4.26, -1.93], 'ATGCA': [4.66, -1.2], 'AGCCC': [-2.95, -1.98], 'ATGCG': [5.09, -0.76], 'AGCCA': [-3.86, -2.44], 'TCGAG': [5.47, -1.2], 'TTCCC': [-0.97, -1.61], 'CTGAG': [4.27, -0.89], 'TTCCA': [-0.69, -0.87], 'TTCCG': [-1.8, -1.46], 'CAGTG': [-1.18, -2.39], 'ACCAG': [-0.64, 3.13], 'ACCAA': [-0.94, 4.34], 'ACCAC': [-0.49, 3.35], 'TAGAG': [-1.35, -0.44], 'TAGAC': [-1.45, -1.45], 'TAGAA': [-1.4, -0.49], 'TATCC': [-4.17, -1.31], 'TATCA': [-3.4, -1.13], 'CTCCG': [-1.83, -0.61], 'TATCG': [-3.35, -0.47], 'TGCTC': [-2.29, -2.44], 'ACGTG': [5.64, -3.13], 'TCCTG': [-1.05, -2.79], 'GTACC': [4.32, -2.37], 'TCCTC': [-1.37, -2.86], 'GTACG': [4.68, -2.11], 'TGCTG': [-1.51, -1.86], 'GCGGG': [2.34, -1.73], 'GCGGC': [2.78, -1.31], 'GCAGG': [2.96, -3.15], 'AGCTG': [-2.49, -2.54], 'TCTCG': [-1.9, -1.21], 'ACGTA': [5.09, -2.68], 'GCAGC': [3.55, -2.09], 'AGCTC': [-3.25, -2.86], 'TCTCC': [-2.23, -2.03], 'AGCTA': [-2.66, -2.32], 'ATGCT': [4.59, -2.2], 'GTTGC': [-2.98, 4.66], 'ACACG': [4.66, -2.25], 'ACACA': [2.96, -2.65], 'ACACC': [3.95, -2.95], 'ATATG': [5.85, -4.26], 'ATATA': [8.32, -2.62], 'ATATC': [6.33, -3.98], 'GAGAG': [-1.93, -1.9], 'GAGAC': [-1.45, -1.85], 'GAATG': [-3.64, -5.91], 'GAATC': [-3.18, -5.55], 'ATTTA': [-2.36, -0.94], 'ATTTC': [-4.4, -3.13], 'GGGAG': [-1.0, -1.87], 'ATTTG': [-4.68, -3.54], 'ACTCA': [-2.78, -1.75], 'TGTGG': [-1.82, 2.59], 'ACTCC': [-2.84, -2.04], 'TGTGA': [-2.33, 3.97], 'TGTGC': [-1.85, 3.31], 'ACTCG': [-1.9, -1.04], 'TATGG': [-2.84, 3.24], 'GGCGG': [-2.02, 1.79], 'TATGC': [-2.97, 5.2], 'GGCGC': [-1.71, 2.85], 'CAATG': [-2.63, -4.37], 'TAAAG': [-1.56, -3.87], 'TAAAA': [-2.46, -2.75], 'TAAAC': [-1.29, -2.84], 'CACCG': [-2.48, -0.28], 'GATGC': [-4.24, 3.67], 'GATGG': [-3.42, 4.13], 'AAGCG': [-2.38, -2.72], 'AAGCA': [-3.05, -3.31], 'AAGCC': [-3.7, -4.34], 'GCTAG': [-2.24, 4.03], 'AAGCT': [-3.24, -4.35], 'AACTG': [-3.28, -1.85], 'AACTA': [-4.24, -3.9], 'AACTC': [-4.38, -3.01], 'TTCGG': [-0.53, 2.76], 'GTTCG': [-3.31, -0.26], 'TTCGC': [-0.23, 3.79], 'TTCGA': [0.19, 5.38], 'CGTGG': [-2.34, 3.23], 'GTTCC': [-3.78, -1.77], 'TGTCC': [-2.88, -2.2], 'TGTCG': [-2.39, -1.7], 'TACGG': [-2.19, 2.74], 'TACGA': [-1.92, 4.91], 'TACGC': [-2.07, 4.31], 'AGTAG': [-4.12, 4.64], 'CAAAG': [-2.8, -4.84], 'AGTAC': [-2.73, 4.94], 'AGTAA': [-3.91, 5.33], 'AGGCC': [-1.89, -2.34], 'AGGCA': [-2.06, -2.41], 'AGGCG': [-2.1, -2.16], 'GCGCG': [3.12, -1.15], 'CATGG': [-4.03, 3.95], 'GGAAG': [-0.61, -3.81], 'TCTGC': [-1.2, 4.1], 'TCTGG': [-2.71, 2.85], 'AGGCT': [-2.71, -3.85], 'ACAGA': [3.72, -1.93], 'ACAGC': [5.07, -2.1], 'ACAGG': [4.68, -2.48], 'ATTAC': [-2.51, 5.03], 'ATTAA': [-2.12, 8.64], 'ATTAG': [-1.62, 7.71], 'ACAGT': [4.57, -1.83], 'AATAA': [-3.91, 8.21], 'AATAC': [-4.27, 6.24], 'AATAG': [-5.98, 3.47], 'TCCCG': [-0.91, -1.93], 'ACGGT': [3.68, -1.2], 'TCCCC': [-1.23, -2.33], 'ACGGC': [3.91, -1.58], 'ACGGA': [2.56, -2.1], 'CTCGG': [-0.49, 4.6], 'AATAT': [-5.48, 6.33], 'TTGAG': [5.89, -0.68], 'GTCAG': [-1.29, 4.57], 'TGATC': [-1.37, -4.67], 'TGATG': [-0.46, -3.98], 'GGCCG': [-2.0, -1.3], 'TTGAA': [5.46, 0.18], 'TTGAC': [6.83, -0.77], 'GTTTG': [-3.27, -2.42], 'ACGGG': [2.76, -2.05], 'TCGGC': [3.91, -0.99], 'TCGGA': [3.14, -0.81], 'CGACG': [-1.47, -2.51], 'AAGGA': [-3.38, -1.91], 'AAGGC': [-2.75, -2.62], 'TTTGG': [-3.1, 2.51], 'TTTGA': [-2.67, 5.24], 'AAGGG': [-3.37, -3.54], 'TTTGC': [-3.68, 3.96], 'AAGGT': [-4.81, -3.27], 'AATTC': [-6.49, -3.59], 'AATTA': [-5.21, -2.38], 'AATTG': [-5.32, -2.97], 'AAAGT': [-6.36, -4.78], 'GACGC': [-2.45, 4.23], 'GACGG': [-2.46, 3.11], 'AAAGC': [-5.07, -3.95], 'TCGTC': [4.81, -2.39], 'AAAGA': [-4.76, -3.21], 'AAAGG': [-4.8, -5.06], 'GGTAG': [-2.63, 4.24], 'ACGCG': [4.51, -1.39], 'TGCAC': [-0.76, 2.85], 'ACGCC': [3.67, -2.09], 'ACGCA': [4.21, -1.26], 'TGCAG': [-0.7, 3.43], 'TAGTA': [-1.21, -2.12], 'TCGTG': [5.8, -2.18], 'TAGTC': [-1.0, -2.07], 'TAGTG': [-1.38, -2.73], 'GCATG': [4.69, -3.87], 'TACCA': [-2.45, -0.68], 'TACCC': [-3.44, -1.79], 'TACCG': [-2.35, -0.83], 'AGAGG': [-1.11, -2.6], 'AGGGT': [-3.17, -2.05], 'AGAGA': [-1.55, -1.87], 'AGAGC': [-1.41, -2.48], 'AGGGG': [-2.55, -2.41], 'AGAGT': [-0.84, -1.51], 'AGGGC': [-2.27, -2.16], 'AGGGA': [-2.51, -1.61], 'CATCG': [-4.47, -0.95], 'CACGG': [-2.43, 4.06], 'TTCTC': [-1.63, -2.79], 'TTCTG': [-0.54, -1.9], 'TCCGC': [-0.89, 2.4], 'TCCGG': [-0.62, 1.31], 'ATCAG': [-0.36, 4.36], 'ATCAA': [0.04, 6.62], 'ATCAC': [-0.51, 3.72], 'AAACG': [-2.94, -3.27], 'CTACG': [4.93, -2.16], 'AAACC': [-3.75, -4.65], 'ACGTC': [5.05, -3.06], 'AAACA': [-2.98, -2.89], 'ACATA': [4.85, -4.73], 'AAACT': [-4.3, -5.49], 'GTTGG': [-1.94, 6.57], 'GCAAG': [4.47, -3.39], 'CGGGG': [-1.77, -1.76], 'CGAGG': [-1.26, -2.8], 'TTTCA': [-1.64, 0.49], 'TTTCC': [-3.36, -1.41], 'GGTTG': [-3.18, -2.06], 'CAGAG': [-1.81, -1.0], 'TTTCG': [-2.61, -0.38], 'TCAAC': [5.39, -3.34], 'TCAAG': [5.51, -3.44], 'ATAGG': [5.54, -2.45], 'ATAGA': [6.34, -1.51], 'ATAGC': [4.96, -2.59], 'ATAGT': [3.79, -3.25], 'TATTC': [-4.85, -3.1], 'TATTG': [-3.9, -2.87], 'GACCG': [-2.77, -0.84], 'GACCC': [-3.44, -1.78]}
	complimentMatrix = {'A':'T', 'T':'A',
				  		'G':'C', 'C':'G'}

	rolls = np.array(rollMapping.values()).flatten()
	maxRoll = max(rolls)
	minRoll = min(rolls)

	bpRoll = []
	DNAarray = list(sequence.upper())
	previousRoll = []
	compliment = False
	for index in range(len(DNAarray)-4):
		pentamer = ''.join(DNAarray[index:index+5])

		if pentamer not in rollMapping.keys():
			compliment = True
			pentamerCompliment = ''

			for bp in reversed(DNAarray[index:index+5]):
				pentamerCompliment = pentamerCompliment + complimentMatrix[bp]

			pentamer = pentamerCompliment

		roll = copy.copy(rollMapping[pentamer])
		if compliment:
			roll.reverse()

		if index == 0:
			bpRoll.append(roll[0])
		else:
			bpRoll.append(np.mean([roll[0],
								   previousRoll[1]]))
		if index == len(DNAarray)-5:
			bpRoll.append(roll[1])

		previousRoll = roll
		compliment = False

	# running average of window = 3
	weights = np.repeat(1.0, window)/float(window)
	runningAverage = np.convolve(bpRoll, weights, 'valid')

	# normalizing data
	normalize = (ymax - ymin)*(runningAverage - minRoll)/(maxRoll - minRoll) + ymin;

	# pulling out feature indecies
	# helix calculation adds one digit, original length list = 11, new list = 8
	middle = len(runningAverage)/2 + len(runningAverage)%2
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	return bpRoll, runningAverage[start-overhang:end+overhang], normalize[start-overhang:end+overhang]

def dinucleotide(sequence):
	"""
	Takes a DNA sequence and converts it into a list of floats based on the dinucleotide pairs
	where G = 0010, C = 0.5, T = -0.5, A = -1
	"""
	frog = []

	for i in range(0,(len(sequence)-1)):
		bp = sequence[i]
		bp_next = sequence[i+1]
		bp = bp.capitalize()
		bp_next = bp_next.capitalize()

		if bp == 'A':
			if bp_next == 'A':
				frog.append([-1,-1,-1,-1])
			elif bp_next == 'C':
				frog.append([-1,-1,-1,1])
			elif bp_next == 'G':
				frog.append([-1,-1,1,-1])
			elif bp_next == 'T':
				frog.append([-1,-1,1,1])
		elif bp == 'C':
			if bp_next == 'A':
				frog.append([-1,1,-1,-1])
			elif bp_next == 'C':
				frog.append([-1,1,-1,1])
			elif bp_next == 'G':
				frog.append([-1,1,1,-1])
			elif bp_next == 'T':
				frog.append([-1,1,1,1])
		elif bp == 'G':
			if bp_next == 'A':
				frog.append([1,-1,-1,-1])
			elif bp_next == 'C':
				frog.append([1,-1,-1,1])
			elif bp_next == 'G':
				frog.append([1,-1,1,-1])
			elif bp_next == 'T':
				frog.append([1,-1,1,1])
		elif bp == 'T':
			if bp_next == 'A':
				frog.append([1,1,-1,-1])
			elif bp_next == 'C':
				frog.append([1,1,-1,1])
			elif bp_next == 'G':
				frog.append([1,1,1,-1])
			elif bp_next == 'T':
				frog.append([1,1,1,1])
	frog = np.array(frog).flatten()

	return frog

def reverseCompliment(sequence):
	"""
	Returns the reverse compliment of the DNA sequence
	"""
	complimentMatrix = {'A':'T', 'T':'A',
				  		'G':'C', 'C':'G'}
	complimentArray = []
	DNAarray = list(sequence.upper())
	for bp in reversed(DNAarray):
		complimentArray.append(complimentMatrix[bp])

	compliment = ''.join(complimentArray)
	return compliment


def DNA_encoding(sequence, recLength = 5, overhang = 0):
	"""
	Takes a DNA sequence and converts it into a list of floats
	where G = [-1,1], C = [1,-1], T = [1,1], A = [-1,-1]
	"""
	encoding = []

	for bp in sequence:
		bp = bp.capitalize()
		if bp == 'A':
			encoding.append([-1,-1])
		elif bp == 'C':
			encoding.append([1,-1])
		elif bp == 'G':
			encoding.append([-1,1])
		elif bp == 'T':
			encoding.append([1,1])

	# pulling out feature indecies
	middle = len(encoding)/2 + len(encoding)%2 - 1
	start = middle - recLength/2
	end = middle + recLength/2 + 1

	encoding = np.array(encoding[start-overhang:end+overhang]).flatten()

	return encoding

