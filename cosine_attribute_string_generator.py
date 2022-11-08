string = "["
for i in range(54):
    string += f"\"COSINE{i}\", "
string=string[0:-2]
string+="]"
print(string)