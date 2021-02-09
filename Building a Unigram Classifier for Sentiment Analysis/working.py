import re
import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#global variables
reg_front_quote = re.compile(r'(\s\'(\w+)\s)')
reg_back_quote = re.compile(r'(\s(\w+)\'\s)')
reg_front_back_quote = re.compile(r'(\s\'(\w+)\'\s)')
reg_negation = re.compile(r'(\w+n\'t$|\bnot\b|\bno\b|\bnever\b|\bcannot\b)')
negation_ending_tokens = ["but","however","nevertheless",".","?","!"]
#is this a global variable
vocabulary_dict = {}

def load_corpus(corpus_path):
	file = open(corpus_path,"r")
	line = file.readline()
	list_of_snippets = []
	while line :
		parts_of_line = line.split("\t")
		new_snippet = [str(parts_of_line[0]),int(parts_of_line[1])]
		new_snippet = tuple(new_snippet)
		list_of_snippets.append(new_snippet)
		line = file.readline()
	return list_of_snippets

def tokenize(snippet):
	snippet = reg_front_quote.sub(replace,snippet)
	snippet = reg_back_quote.sub(r" \g<2> ' ",snippet)
	snippet = reg_front_back_quote.sub(r" ' \g<2> ' ",snippet)
	tokenized_list = snippet.split()
	return tokenized_list

def tag_edits(tokenized_snippet):
	edit_flag = False
	for index,token in enumerate(tokenized_snippet) :
		#check if the token contains an open square bracket
		if(token.find("[") != -1) :
			edit_flag = True
			tokenized_snippet[index] = tokenized_snippet[index].strip("[")
		#check if edit flag is true and tag the contents
		if(edit_flag):
			tokenized_snippet[index] = "EDIT_"+tokenized_snippet[index]
		#check if the token contains a closed square bracket 
		if(token.find("]") != -1):
			edit_flag = False
			tokenized_snippet[index] = tokenized_snippet[index].strip("]")
	return tokenized_snippet

def tag_negation(tokenized_snippet) :
	copy_snippet = tokenized_snippet[:]
	removed_token_list = []
	for index,token in enumerate(copy_snippet) :
		#checking if meta tag exists and removing it
		if(token.find("EDIT_") != -1) :
			copy_snippet[index] =copy_snippet[index].strip("EDIT_")
			if(copy_snippet[index] == "") :
				del copy_snippet[index]
				del tokenized_snippet[index]			
	# print(copy_snippet)
	pos_tags_list = nltk.pos_tag(copy_snippet)
	for index,(word,pos) in enumerate(pos_tags_list) :
		if(tokenized_snippet[index].find("EDIT_") != -1) :
			pos_tags_list[index] = ("EDIT_"+word,pos)
	negation_flag = False
	#checking for negation words
	for index,(word,pos) in enumerate(pos_tags_list) :
		if(word in negation_ending_tokens or pos in ["JJR","RBR"]) :
			negation_flag = False
		if(negation_flag) :
			pos_tags_list[index] = ("NOT_"+word,pos)
		if (reg_negation.search(word) != None) :
			if(word == "not" and index+1 < len(pos_tags_list)  and pos_tags_list[index+1][0] == "only") :
				negation_flag = False
			else :
				negation_flag = True

	return pos_tags_list

def replace(match_object) :
	not_replaceable = ["70s","em","twould","tis"]
	if(match_object.group(2) in not_replaceable) :
		return match_object.group(1)
	else :
		return " ' "+match_object.group(2)+" " 

def get_features(preprocessed_snippet) :
	feature_vector = np.zeros(len(vocabulary_dict))
	for index,(word,pos) in enumerate(preprocessed_snippet) :
		if(word.find("EDIT_") == -1):
			word_index = vocabulary_dict.get(word)
			if(word_index != None) :
				feature_vector[word_index] = feature_vector[word_index] + 1
	return feature_vector 

def normalize(X) :
	X_normalized = (X - X.min(axis=0))/ (X.max(axis=0) -X.min(axis=0))
	#checking if there are nan values and converting them to 1
	index_of_NaNs = np.isnan(X_normalized)
	X_normalized[index_of_NaNs] = 0
	return X_normalized

def evaluate_predictions(Y_pred, Y_true) :
	tp = 0.0
	fp = 0.0
	fn = 0.0
	for index,value in enumerate(Y_pred) :
		if(Y_true[index] == 1 and Y_pred[index] == 1) :
			tp = tp + 1
		elif(Y_true[index] == 0 and Y_pred[index] == 1) :
			fp = fp + 1
		elif(Y_true[index] == 1 and Y_pred[index] == 0) :
			fn = fn + 1
	p = tp/(tp+fp)
	r = tp/(tp+fn)
	fmeasure = 2 * (p * r)/(p+r)
	return (p,r,fmeasure)

#this function returns the absolute value of the weight in the (index,weight) element for sorting purposes
def take_absolute_weight(element):
	return abs(element[1])

def get_key_from_vocabulary(index) :
	for key,value in vocabulary_dict.items() :
		if(value == index) :
			return key

def top_features(logreg_model,k) :
	list_top_features = []
	for index,weight in enumerate(logreg_model.coef_[0]) :
		list_top_features.append((index,weight))
	list_top_features.sort(key=take_absolute_weight,reverse=True)
	for index,(word_index,weight) in enumerate(list_top_features) :
		list_top_features[index] =(get_key_from_vocabulary(word_index),weight)
	return list_top_features[:k]

def load_dal(dal_path) :
	file = open(dal_path,"r")
	header_line = file.readline()
	line = file.readline().strip("\n")
	score_dictionary = {}
	while line :
		parts_of_line = line.split("\t")
		score_dictionary[parts_of_line[0]] = tuple(parts_of_line[1:])
		line = file.readline().strip("\n")
	print(score_dictionary["zone"])

if __name__ == '__main__':
	snippet_label_list = load_corpus("train.txt")
	for index, (snippet,label) in enumerate(snippet_label_list) :
		new_snippet = tokenize(snippet)
		new_snippet = tag_edits(new_snippet)
		new_snippet = tag_negation(new_snippet)
		snippet_label_list[index] =(new_snippet,label)
	vocabulary = set()
	for index, (snippet,label) in enumerate(snippet_label_list) :
		for index,(word,pos) in enumerate(snippet) :
			#if the word does not contain edit meta tag then only add it to vocabulary
			if(word.find("EDIT_") == -1):
				vocabulary.add(word)
	vocabulary = list(vocabulary)
	for index,word in enumerate(vocabulary) :
		vocabulary_dict[word] = index

	X_train = np.empty([len(snippet_label_list),len(vocabulary_dict)])
	Y_train = np.empty(len(snippet_label_list))
	for index, (snippet,label) in enumerate(snippet_label_list) :
		feature_vector = get_features(snippet)
		X_train[index] = feature_vector
		Y_train[index] = label
	# print(X_train)
	X_train = normalize(X_train)
	naive_bayes_classifier = GaussianNB()
	logistic_regression_classifier = LogisticRegression()
	# print(vocabulary_dict)
	# print(X_train)
	naive_bayes_classifier.fit(X_train,Y_train)
	logistic_regression_classifier.fit(X_train,Y_train)
	snippet_label_test_list = load_corpus("test.txt")
	for index, (snippet,label) in enumerate(snippet_label_test_list) :
		new_snippet = tokenize(snippet)
		new_snippet = tag_edits(new_snippet)
		new_snippet = tag_negation(new_snippet)
		snippet_label_test_list[index] =(new_snippet,label)
	X_test = np.empty([len(snippet_label_test_list),len(vocabulary_dict)])
	Y_test = np.empty(len(snippet_label_test_list))
	for index, (snippet,label) in enumerate(snippet_label_test_list) :
		feature_vector = get_features(snippet)
		X_test[index] = feature_vector
		Y_test[index] = label
	X_test = normalize(X_test)
	Y_pred_n = naive_bayes_classifier.predict(X_test)
	Y_pred_l = logistic_regression_classifier.predict(X_test)
	print(evaluate_predictions(Y_pred_n,Y_test))
	print(evaluate_predictions(Y_pred_l,Y_test))
	print(top_features(logistic_regression_classifier,10))
	load_dal("dict_of_affect.txt")






	
