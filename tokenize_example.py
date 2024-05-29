from pickle import load
import tiktoken, time

BPE = tiktoken.get_encoding("cl100k_base")
gpt2 = tiktoken.get_encoding("gpt2")

tokenizer = load(open(f".\\model\\tokenizer.pkl", "rb"))

string = "오를레앙의 안나 마리아(Anne-Marie d'Orléans, 1669년 8월 27일 ~ 1728년 8월 26일)는 사르데냐 국왕 비토리오 아메데오 2세의 왕비로, 프랑스식 이름은 안 마리 도를레앙(Anne Marie d'Orléans)이다. 프랑스 국왕 루이 14세의 동생 오를레앙 공 필리프 1세와 그의 첫 번째 공작부인 잉글랜드 공주 헨리에타 앤 사이에서 둘째딸로 태어났다. 루이 14세는 이탈리아에 대한 프랑스의 영향력을 유지하기 위해 조카딸을 사보이 공국의 대공에게 시집보내기로 결정했고 두 사람은 1684년 4월 10일 베르사유에서 결혼했다."


s = time.time()
encoded = tokenizer.encode(string)
# print("token: "+ str(encoded))
print("token: "+ str([tokenizer.decode([c]) for c in encoded]))
print(len(encoded))
print("org: ", len(string))
print(time.time() - s)
s = time.time()
encoded = BPE.encode(string)
print("token: "+ str([BPE.decode([c]) for c in encoded]))
print(len(encoded))
print("org: ", len(string))
print(time.time() - s)
s = time.time()
encoded = gpt2.encode(string)
print("token: "+ str([gpt2.decode([c]) for c in encoded]))
print(len(encoded))
print("org: ", len(string))
print(time.time() - s)