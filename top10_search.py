list1 = ujson.load(open("/Users/jason.wu/Downloads/ap_cfd_dis5_min10_top20_stp.json"))
cfd1 = nltk.ConditionalFreqDist()
for w in list1:
    cfd1[w]=nltk.FreqDist(list1[w])
cfd1["billion"].most_common()
