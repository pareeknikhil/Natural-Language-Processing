import nltk
import re
import sys


# Fill in the pattern (see Part 2 instructions)
NP_grammar = 'NP: {<DT|WDT>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}' 


# Fill in the other 4 rules (see Part 3 instructions)
hearst_patterns = [
    ('((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?include (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?like (NP_\w+ ? (, )?(and |or )?)+)','before'), 
    ('(NP_\w+ (, )?for example (, )?(NP_\w+ ?(, )?(and |or )?)+)','before'), 
    ('(NP_\w+ (, )?for instance (, )?(NP_\w+ ?(, )?(and |or )?)+)','before'),
    ('(NP_\w+ (, )?notably (NP_\w+ ? (, )?(and |or )?)+)','before')] 

# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples
def load_corpus(path):
    file = open(path,"r")
    line = file.readline()
    list_of_lines = []
    while line :
        sentence,lemmatized = line.split("\t")
        sentence = sentence.split()
        sentence = [word.strip() for word in sentence]
        lemmatized = lemmatized.split()
        lemmmatized = [word.strip() for word in lemmatized]
        list_of_lines.append(tuple([sentence,lemmatized]))
        line = file.readline()
    return list_of_lines



# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    file = open(path,"r")
    line = file.readline()
    true_set = set()
    false_set = set()
    while line :
        hyponym,hypernym,label = [word.strip() for word in line.split("\t")]
        if(label == "True") :
            true_set.add(tuple([hyponym,hypernym]))
        else :
            false_set.add(tuple([hyponym,hypernym]))
        line = file.readline()
    return tuple([true_set,false_set])


# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):
    tagged_sentence = nltk.pos_tag(sentence)
    tagged_lemmatized = []
    for index,(token,tag) in enumerate(tagged_sentence) :
        tagged_lemmatized.append(tuple([lemmatized[index],tag]))
    lemmmatized = tagged_lemmatized
    lemmmatized_tree  = parser.parse(lemmmatized)
    list_of_chunks = tree_to_chunks(lemmmatized_tree)
    string_of_chunks = merge_chunks(list_of_chunks)
    return string_of_chunks

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    list_of_chunks = []
    for child in tree :
        if(isinstance(child,nltk.Tree)):
            list_of_tokens = [token for (token,tag) in child]
            list_of_tokens = ('_').join(list_of_tokens)
            list_of_tokens = "NP_" + str(list_of_tokens)
            list_of_chunks.append(list_of_tokens)
        else:
            list_of_chunks.append(child[0])
    return list_of_chunks
    


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    if(len(chunks) > 0) :
        buffer_of_chunks = []
        i = 0
        j = 0
        buffer_of_chunks.append(chunks[0])
        i= i+1
        while(i < len(chunks)):
            if((chunks[i][:3] == "NP_") and (buffer_of_chunks[j][:3] == "NP_") ):
                buffer_of_chunks[j] = buffer_of_chunks[j] + "_" +chunks[i][3:]
                i = i+1
            else :
                buffer_of_chunks.append(chunks[i])
                i=i+1
                j=j+1

        string_of_chunks = (" ").join(buffer_of_chunks)
        return string_of_chunks
    else :
        return " "
# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
    for regex_pattern in hearst_patterns :
        match = re.search(regex_pattern[0],chunked_sentence)
        if (match is not None):
            match = match.group(0)
            list_of_strings = match.split()
            list_of_strings =[string for string in list_of_strings if string[:3] == "NP_"]            
            list_of_strings = postprocess_NPs(list_of_strings)
            if(regex_pattern[1] == "before") :
                hypernym = list_of_strings[0]
                hyponyms = list_of_strings[1:]
            elif(regex_pattern[1] == "after") :
                hypernym = list_of_strings[-1]
                hyponyms = list_of_strings[:-1]
            for hyponym in hyponyms :
                yield tuple([hyponym,hypernym])


# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    for i,NP in enumerate(NPs) :
        NPs[i] = NPs[i].replace("NP_","")
        NPs[i] = NPs[i].replace("_"," ")
    return NPs


# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):
    true_positive = 0
    false_positive = 0
    for extraction in extractions :
        if (extraction in gold_true) :
            true_positive = true_positive + 1
        elif(extraction in gold_false) :
            false_positive  = false_positive + 1 
    precision  = float(true_positive) / float(true_positive + false_positive)
    false_negative = len(gold_true - set(extractions))
    recall = float(true_positive)/float(true_positive+false_negative)
    fmeasure =  float(2 * precision * recall) /float(precision + recall)
    return tuple([precision,recall,fmeasure])

def main(args):
    corpus_path = args[0]
    test_path = args[1]

    wikipedia_corpus = load_corpus(corpus_path)
    test_true, test_false = load_test(test_path)

    NP_chunker = nltk.RegexpParser(NP_grammar)

    # Complete the line (see Part 2 instructions)
    wikipedia_corpus = [chunk_lemmatized_sentence(sentence,lemmmatized,NP_chunker) for (sentence,lemmmatized) in wikipedia_corpus ]

    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)

    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    