from xx import analysis

print("Write:\n")
sentence = input()
var=0
while(sentence):
	
	result = analysis(sentence)
	if result[0]==0:
		print("\nNegative\n")
	else:
		print("\nPositive/Neutral\n")

	print("Was the output faithful?\n")
	print("0->Yes")
	print("1->No\n")
	var=int(input())
	print(" ")
	print("Write:\n")
	sentence = input()


