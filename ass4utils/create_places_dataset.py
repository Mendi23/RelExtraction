input_file = "GeoLite2-City-Locations-en.csv"
output_file = "places.txt"


with open(input_file, encoding="utf8") as inF:
    inF.readline()
    places = set()
    for line in inF:
        line_data = line.split(",")
        for i in [2, 3, 4, 5, 10]:
            places.add(line_data[i].replace("\"", "").replace("-", " ").lower())


with open(output_file, "w", encoding="utf8") as outF:
    for p in places:
        outF.write(p+"\n")

