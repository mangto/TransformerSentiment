import tensorflow as tf
from transformer import *
from pickle import load
from threading import Thread

tokenizer = load(open(f".\\model\\tokenizer.pkl", "rb"))

START_TOKEN, END_TOKEN, VOCAB_SIZE = SEV(tokenizer)

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(f".\\model2\\model.ckpt")

from threading import Thread
import pygame, sys

window = pygame.display.set_mode((512, 512))
pygame.display.set_caption('감정 분석')


table = {
    '공포':(255, 0, 132),
    '놀람':(255, 255, 255),
    '분노':(255, 0, 0),
    '슬픔':(0, 138, 255),
    '중립':(0, 255, 96),
    '행복':(255, 216, 0),
    '혐오':(132, 255, 0),
}
color = (0, 0, 0)

def get_user():
    
    global color
    while True:
        user = input(" >>> ")
        out = predict(user, tokenizer, model, START_TOKEN, END_TOKEN, MAX_LENGTH)
        print("token: "+ str(tokenizer.encode(user)))
        print("token: "+ str([tokenizer.decode([c]) for c in tokenizer.encode(user)]))
        
        print("prediction: " + out)
        color = table.get(out,(0, 0, 0))

Thread(target=get_user).start()

while True:
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()
            sys.exit()
    window.fill(color)
    pygame.display.update()