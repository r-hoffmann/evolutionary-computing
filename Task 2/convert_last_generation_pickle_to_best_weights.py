import pickle

dir_name = '51_54'
with open(dir_name + '/last_generation.pkl', 'rb') as fid:
    data = pickle.load(fid)

best = data['best']
list_of_weights = []
for node in best.nodes:
    if best.nodes[node].key > 5:
        list_of_weights.append(best.nodes[node].bias)
    print(best.nodes[node])

for conn in best.connections:
    if best.connections[conn].key[0] < 0:
        list_of_weights.append(best.connections[conn].weight)

for node in best.nodes:
    if best.nodes[node].key < 5:
        list_of_weights.append(best.nodes[node].bias)

for conn in best.connections:
    if -1 < best.connections[conn].key[1] < 5:
        list_of_weights.append(best.connections[conn].weight)

print(len(list_of_weights))

with open(dir_name + '/21.txt', 'w') as fid:
    for w in list_of_weights:
        fid.write(str(w) + '\n')
