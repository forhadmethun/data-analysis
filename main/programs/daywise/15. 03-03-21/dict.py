
import json

list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS", "Reconnaissance", "Normal"]
dict = {

}
for i in range(0, 10):
    dict[i] = {"Fuzzers": 0, "Exploits": 0, "Worms": 0, "Shellcode": 0, "Generic": 0, "Analysis": 0, "Backdoor": 0, "DoS": 0,
         "Reconnaissance": 0, "Normal": 0}

# print(dict)
print(json.dumps(dict, indent=2))
# print(len(list))