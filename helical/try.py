import json
import codecs
file = "/Users/marco/Downloads/test_pyramidal/200104066_09_SFOG.mrxs.gson"

# reader = codecs.getreader("utf-8")
# obj = json.load(reader(file))


with open(file, 'rb') as f:
    # text = json.load(file)
    text = f.read()[7:]



json_text = json.dumps(text.decode(), indent = 4)
print(json_text)

save_fp = file.replace('gson', 'json')
with open(save_fp, 'w') as f:
    # text = json.load(file)
    f.write(json_text)

with open(save_fp, 'r') as f:
    text = json.load(save_fp)

print(text[0])
