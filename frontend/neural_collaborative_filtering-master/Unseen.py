def getList(num_range, list):
    unseen = []
    templist = list
    print(templist.__len__())
    for i in range(num_range):
        if len(templist)>0 and i == templist[0]:
            del templist[0]
            continue
        unseen.append(i)
    print unseen.count(593)
    return unseen

