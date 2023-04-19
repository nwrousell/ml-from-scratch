chunk_size = (2 ** 10) * 8 # 8 kilobytes

with open("data/test-labels.gz", mode="rb") as labels_file:
    contents = labels_file.read(32)

# print(type(contents))
print(contents)