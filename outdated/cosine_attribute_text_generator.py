i=0
text = "\"@attribute COSINE0 numeric\""

for i in range(1,54):
    text=text+f",\n\"@attribute COSINE{i} numeric\""

print(text)
