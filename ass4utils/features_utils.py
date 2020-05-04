
verbose = True


def readVectorsFromFile(fileName):
    words = {}
    if verbose: print("Reading", fileName)
    with open(fileName, "r") as lines:
        for line in lines:
            vector = line.split()
            word = vector.pop(0)
            words[word] = list(map(float, vector))
    return words
