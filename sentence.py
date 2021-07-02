
sentids = ['36680951', '9864956']
sent = {}
with open('D:/SemMed/sentence.csv', 'r') as fin:
    count = 0
    for line in fin:
        count += 1
        if count % 1000000 == 0:
            print(count)
        ls = line.strip().split(',')
        if ls[0] in sentids:
            sent[ls[0]] = ''.join(ls[8:])
            print(ls)
        if len(sent) == len(sentids):
            break
print(sent)