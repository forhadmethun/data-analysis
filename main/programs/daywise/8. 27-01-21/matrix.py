attack_categories = ['Analysis', 'Shellcode', 'Fuzzers', 'Generic', 'DoS', 'Worms', 'Reconnaissance', 'Normal', 'Exploits', 'Backdoor']

print('%15s' % '', end= " ")
for i in range(0, len(attack_categories)):
    print('%15s' % attack_categories[i], end= " ")
print()

cols_count = len(attack_categories)
rows_count = cols_count
result_matrix = [[0 for x in range(cols_count)] for x in range(rows_count)]

for i in range(0, len(attack_categories)):
    mat_elem = {}
    for j in range(i+1, len(attack_categories)):
        result_matrix[i][j] = str(i) + " " + str(j)
        # mat[attack_categories]

for i in range(0, len(attack_categories)):
    print('%15s' % attack_categories[i], end=" ")
    for j in range(0, len(attack_categories)):
        if( result_matrix[i][j] == 0):
            print('%15s' % "-", end = " ")
        else:
            print('%15s' % (result_matrix[i][j]), end = " ")
    print()
print(attack_categories)