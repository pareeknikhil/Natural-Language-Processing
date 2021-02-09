import numpy
import scipy
import sklearn
from nltk.corpus import brown
import re
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import nltk

#global variables
tag_dict = {}
feature_dict = {}
rare_words = set()


def main() :
	brown_sentences = brown.tagged_sents(tagset='universal')
	train_sentences = []
	train_tags = []
	for sentence in brown_sentences :
		sentence_words = []
		sentence_tags = []
		for index,(word,pos) in enumerate(sentence):
			sentence_words.append(word)
			sentence_tags.append(pos)
		train_sentences.append(sentence_words)
		train_tags.append(sentence_tags)
	vocabulary_dict = {}
	for sentence in train_sentences :
		for word in sentence :
			if(vocabulary_dict.get(word) is None) :
				vocabulary_dict[word] = 1
			else :
				vocabulary_dict[word] = vocabulary_dict[word] + 1
	for (word,count) in vocabulary_dict.items() :
		if(count<5) :
			rare_words.add(word)
	training_features = [[]]

	for sentence_index,sentence in enumerate(train_sentences):
		training_features.append([])
		for word_index,word in enumerate(sentence):
			if(word_index == 0) :
				prevtag = '<S>'
			else :
				prevtag = train_tags[sentence_index][word_index-1]
			training_features[sentence_index].append(get_features(word_index, sentence, prevtag, rare_words))
	
	training_features,non_rare_features = remove_rare_features(training_features,5)
	counter = 0
	for feature in non_rare_features :
		feature_dict[feature] = counter
		counter = counter+1
	tagset = set()
	
	for sentence in train_tags:
		for tag in sentence :
			tagset.add(tag)
	
	counter = 0
	for tag in tagset :
		tag_dict[tag] = counter
		counter = counter + 1  
	
	Y_train = build_Y(train_tags)
	X_train = build_X(training_features)
	model = LogisticRegression(class_weight='balanced',solver='saga',multi_class='multinomial')
	model.fit(X_train,Y_train)
	
	
	test_data =load_test("test.txt")
	for sentence in test_data :
		temp_data = []
		temp_data.append(sentence)
		Y_pred, Y_start = get_predictions(temp_data, model)
		print(viterbi(Y_start, Y_pred))


def word_ngram_features(i,words):
	if(i > 0) :
		prevbigram_feature = words[i-1].lower()
	else :
		prevbigram_feature = '<s>'
	prevbigram = 'prevbigram-'+prevbigram_feature
	if(i < (len(words)-1)) :
		nextbigram_feature = words[i+1].lower()
	else :
		nextbigram_feature = "</s>"
	nextbigram = 'nextbigram-'+nextbigram_feature
	if(i < 2):
		prevskip_feature = "<s>"
	else :
		prevskip_feature = words[i-2].lower()
	prevskip = 'prevskip-'+ prevskip_feature
	if(i < (len(words)-2)) :
		nextskip_feature = words[i+2].lower()
	else :
		nextskip_feature = "</s>"
	nextskip = 'nextskip-'+nextskip_feature
	prevtrigram = "prevtrigram-"+prevbigram_feature+"-"+prevskip_feature
	nexttrigram = "nexttrigram-"+nextbigram_feature+"-"+nextskip_feature
	centertrigram = "centertrigram-"+prevbigram_feature+"-"+nextbigram_feature
	ngram_features = [prevbigram,nextbigram,prevskip,nextskip,prevtrigram,nexttrigram,centertrigram]
	return ngram_features

def word_features(word,rare_words) :
	features  = []
	if(word not in rare_words):
		features.append("word-"+word.lower())
	if(word[0].isupper()):
		features.append('capital')
	if(re.search(r'-',word) != None) :
		features.append('hyphen')
	if(re.search(r'/d',word) != None) :
		features.append('number')
	for i in range(1,5) :
		if(i<=(len(word))) :
			features.append("prefix"+str(i)+"-"+word.lower()[:i])
	for i in range(-1,-5,-1) :
		if(-i<=(len(word))):
			features.append("suffix"+str(-i)+"-"+word.lower()[i:])			
	return features

def get_features(i, words, prevtag, rare_words) :
	ngram_features = word_ngram_features(i,words)
	other_features = word_features(words[i],rare_words)
	all_features = ngram_features + other_features
	all_features.append("tagbigram-"+prevtag)
	#new features
	all_features.append("word-"+str(words[i])+"-prevtag-"+str(prevtag))
	if(words[i].isupper()) :
		all_features.append('allcaps')
	wordshape = ''
	for letter in words[i] :
		if(letter.isupper()) :
			wordshape+='X'
		elif(letter.islower()) :
			wordshape+='x'
		elif(letter.isdigit()) :
			wordshape+='d'
		else :
			wordshape+=letter
	all_features.append('wordshape-'+str(wordshape))
	shortwordshape = ''
	index = -1
	for i,letter in enumerate(wordshape):
		if(i==0) :
			shortwordshape+=letter
			index = index + 1
		elif (letter != shortwordshape[index]) :
			shortwordshape +=letter
			index=index+1
	all_features.append('short-wordshape-'+str(shortwordshape))
	if(('allcaps' in all_features) and ('digit' in all_features) and ('hyphen' in all_features)):
		all_features.append('allcaps-digit-hyphen') 
	followed_feature = False
	if('capital' in all_features):
		for follow_word in words[i:i+4] :
			if(follow_word in ["Co.","Inc."]) :
				followed_feature = True
	
	if(followed_feature) :
		all_features.append('capital-followedby-co')
	
	return all_features

def remove_rare_features(features, n) :
	feature_dict = {}
	rare_features = set()
	non_rare_features = set()
	for sentence in features :
		for word_features in sentence :
			for feature in word_features :
				if(feature_dict.get(feature) != None) :
					feature_dict[feature] = feature_dict[feature] + 1
				else :
					feature_dict[feature] = 1
	for (word,count) in feature_dict.items() :
		if(count<n) :
			rare_features.add(word)
		else :
			non_rare_features.add(word)

	for sentence_index,sentence in enumerate(features) :
		for word_index,word_features in enumerate(sentence) :
			features[sentence_index][word_index] = [ elem for elem in features[sentence_index][word_index] if elem not in rare_features]
	return features, non_rare_features

def build_Y(tags) :
	Y = []
	for sentence_tags in tags :
		for word_tag in sentence_tags :
			Y.append(tag_dict[word_tag])
	Y = numpy.array(Y)
	return Y

def build_X(features) :
	examples = []
	feature = []
	i = 0
	for sentence_features in features :
		for word_features in sentence_features :
			for individual_feature in word_features :				
				if(feature_dict.get(individual_feature) is not None) :
					examples.append(i)
					feature.append(feature_dict[individual_feature])
			i = i+1
	values = numpy.ones(len(examples),dtype = int)
	values - numpy.array(values)
	examples = numpy.array(examples)
	feature = numpy.array(feature)
	X = csr_matrix((values,(examples,feature)),shape=(i,len(feature_dict)))
	return X

def load_test(filename) :
	test_file = open(filename,"r")
	line = test_file.readline()
	test_data = []
	while(line) :
		line = line.strip()
		line_data = line.split()
		line_data = [ elem.strip() for elem in line_data]
		test_data.append(line_data)
		line = test_file.readline()
	return test_data

def get_predictions(test_sentence, model) :
	Y_pred = numpy.empty([len(test_sentence[0])-1,len(tag_dict),len(tag_dict)]) 
	for i in range(1,len(test_sentence[0])) :
		for (tag,j) in tag_dict.items() :
			features = [[]]
			features[0].append(get_features(i, test_sentence[0], tag, rare_words))
			X = build_X(features)
			Y_pred[i-1,j] = model.predict_log_proba(X)
	start_features = [[]]
	start_features[0].append(get_features(0, test_sentence[0], '<S>', rare_words))
	X = build_X(start_features)
	Y_start= model.predict_log_proba(X)
	return Y_pred,Y_start

def viterbi(Y_start, Y_pred) :
	V = numpy.empty([Y_pred.shape[0]+1,len(tag_dict)])
	BP = numpy.empty([Y_pred.shape[0]+1,len(tag_dict)])
	for j in range(len(tag_dict)):
		V[0,j] = Y_start[0,j]
		BP[0,j] = -1

	for i in range(Y_pred.shape[0]) :
		for (tag_k,k) in tag_dict.items() :
			list_values = numpy.empty([1,len(tag_dict)])
			for (tag_j,j) in tag_dict.items() :
				value = V[i,j] + Y_pred[i,j,k]
				list_values[0,j] = value
			V[i+1,k] = list_values.max()
			BP[i+1,k] = numpy.argmax(list_values)

	backward_indices = []
	n =  int(V.shape[0]-1)
	index =int(numpy.argmax(V[n]))
	backward_indices.append(index)

	while(n>=1) :
		index = int(BP[n,index])
		backward_indices.append(index)
		n = n-1
	
	backward_indices.reverse()
	temp_list = []
	for index in backward_indices :
		for (tag,i) in tag_dict.items() :
			if(i==index):
				temp_list.append(tag)
	return temp_list

if __name__ == '__main__':
	main()