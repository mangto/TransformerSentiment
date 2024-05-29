with open(".\\dataset\\dataset_raw.csv", 'r', encoding='utf8') as file:
    dataset = file.read().splitlines()

text = ''

for line in dataset:
    line = line.replace(',', '.')
    line = line.replace('	', ',')
    # line = line.replace('공포', '0')
    # line = line.replace('놀람', '1')
    # line = line.replace('분노', '2')
    # line = line.replace('슬픔', '3')
    # line = line.replace('중립', '4')
    # line = line.replace('행복', '5')
    # line = line.replace('혐오', '6')

    text += '\n' + line
    continue

with open(".\\dataset\\dataset.csv", 'w', encoding='utf8') as file:
    file.write(text[1:])