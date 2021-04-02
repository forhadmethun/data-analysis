def percentage_increase(a):
    for i in range(1, len(a)):
        # per = (a[i] - a[i - 1]) / a[i - 1]
        per = (a[i] - a[0]) / a[0]
        print(per * 100)


a = [46, 48, 50, 55]
percentage_increase(a)
