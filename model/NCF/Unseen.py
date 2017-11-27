def getList(num_range, list):
    unseen = []
    for i in range(num_range):
        if len(list)>0 and i==list[0]:
            del list[0]
            continue
        unseen.append(i)
    return unseen

if __name__ == '__main__':
    list = [2,4]
    print(getList(5, list))