import re

word = "eEkuuhjhefkj"
prefixes =[]
suffixes = []
# print(re.search(r'-',string1))
for i in range(1,5) :
	print(i)
	if(i<=len(word)) :
		prefixes.append([i,word.lower()[:i]])
for i in range(-1,-5,-1) :
	if(-i<=len(word)):
		suffixes.append([-i,word.lower()[i:]])
list1 = prefixes + suffixes

# dict1[[1,2]] = "hi"
# set1 ={"a","b","c"}
# list1 = ["a","b","c","d","e","f"]
# list1 = [ elem for elem in list1 if elem not in set1]
word='IAAnc1234....$'
wordshape = ''
for letter in word :
	if(letter.isupper()) :
		wordshape+='X'
	elif(letter.islower()) :
		wordshape+='x'
	elif(letter.isdigit()) :
		wordshape+='d'
	else :
		wordshape+=letter

print(wordshape)
shortwordshape = ''
index = -1
for i,letter in enumerate(wordshape):
	if(i==0) :
		shortwordshape+=letter
		index = index + 1
	elif (letter != shortwordshape[index]) :
		shortwordshape +=letter
		index=index+1
print(shortwordshape)

if(set(['a','b']).issubset(['a','b','c'])):
	print('works')