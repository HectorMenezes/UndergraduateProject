import random
lines = open('data/data_banknote_authentication.txt').readlines()
random.shuffle(lines)
open('new.txt', 'w').writelines(lines)