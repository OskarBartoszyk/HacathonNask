from anonimizer import ReadText

# Wczytaj i zanonimizuj jedną linijką
text = ReadText("/Users/oskar/Desktop/HacknationNask/data/anonymized.txt")
result = text.FakeFillAll()
print(result)