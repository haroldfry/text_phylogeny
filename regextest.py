import numpy as np


str = "1 1:279 2:286 3:565 4:281 5:288 6:569 7:148 8:149 9:366 10:412 11:778 12:517 13:0.845161290322581 " \
      "14:0.766159695817491 15:0.669213483146067 16:0.811087298311398 17:0.825132054951516"

testlist = str.split()
del testlist[0]
newlist = []
for element in testlist:
    newele = element.split(":", 1)[-1]
    if newele.find('.'):
        float(newele)
        newlist.append(newele)
    else:
        int(newele)
        newlist.append(newele)

newarray = np.asarray(newlist, dtype=float)
value = np.dot(newarray, np.transpose(newarray))

print value