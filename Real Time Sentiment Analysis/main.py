from pack import analysis
from add_row 

print("Write:\n")
sentence = input()
var=0
while(sentence):
	result = analysis(sentence)
	if result[0]==0:
		print("\nNegative\n")
	else:
		print("\nPositive/Neutral\n")

	print("Was the output faithful?\n Help Create Better Model")
	print("0->Yes")
	print("1->No\n")
	#HERE APPEND THE NEW ROw
	var=int(input())
	if var==0:
		df_new = df.append({"Review" : sentence, "Liked" : result[0]}, ignore_index = True)
	else: 
		if result[0]==1:
			df_new = df.append({"Review" : sentence, "Liked" : 0}, ignore_index = True)
		else:
			df_new = df.append({"Review" : sentence, "Liked" : 1}, ignore_index = True)	
	
	df_new.to_pickle("new_pkl.pkl")
	df = pd.read_pickle("new_pkl.pkl")
	print("\nThank you! Press Enter to Exit or Continue:")
	print("Write:\n")
	sentence = input()


