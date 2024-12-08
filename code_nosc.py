from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist, euclidean
import numpy as np

## ----------------
# Data Preparation 
## ----------------
data = np.loadtxt('diabetic_binary.csv', delimiter=',')
n = data.shape[0]
n_train = int(0.5 * n)
n_valid = int(0.2 * n)
n_test = int(0.25 * n)
x_train = data[0:n_train, 0:-1]
y_train = data[0:n_train, -1]
x_valid = data[n_train:n_train + n_valid, 0:-1]
y_valid = data[n_train:n_train + n_valid, -1]
x_test = data[n - n_test:, 0:-1]
y_test = data[n - n_test:, -1]

# Gamms is used to indirectly control compression rate 
# See Table 2 in the paper for how we set it 
gamma = 6

# Other Preparations 
label1 = 1
label2 = 0
iterations = 2

# Algorithm 1 : Brute-Force net construction
def d(p, S_gamma):
    v1 = p[:-1].reshape((1, -1))  # Exclude the label, reshape p to be 2D (1 sample, n_features)
    v2 = S_gamma[:, :-1]  # Extract features only from S_gamma (exclude labels)
    min_distance = np.min(cdist(v1, v2))  # Compute the minimum distance
    return min_distance

# Construct net  
def net_construction(S):
    S_gamma = S[np.random.choice(S.shape[0], size=1), :]  # Ensure it's 2D
    for i in range(S.shape[0]):
        p = S[i]
        if d(p, S_gamma) >= gamma:
            S_gamma = np.vstack((S_gamma, p.reshape(1, -1)))  # Append the new point
    return S_gamma

Accuracy_method_net = []
Compression_method_net = []
Accuracy_method_heuristic = []
Compression_method_heuristic = []

# Algorithm 2 : Consistent Pruning heuristic
idx_to_remove = set()
for it in range(iterations):
    
    # Construct net on the training data
    S_train = np.column_stack((x_train, y_train))
    s_gamma = net_construction(S_train)
    
    # Net results
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(s_gamma[:, :-1], s_gamma[:, -1])
    pred = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]  # Get probabilities for positive class
    acc = accuracy_score(y_test, pred)
    
    Accuracy_method_net.append(acc)
    Compression_method_net.append(s_gamma.shape[0] / S_train.shape[0])

    # Heuristic results
    for i in range(int(np.abs(np.ceil(np.log(gamma)) + 1))):
        for j in range(s_gamma.shape[0]):
            p = s_gamma[j]
            if j in idx_to_remove:
                continue
            if np.max((p == s_gamma).sum(axis=1)) == s_gamma.shape[1] and d(p, s_gamma) >= 2 * 2 ** i:
                for k in range(s_gamma.shape[0]):
                    q = s_gamma[k]
                    if j != k and euclidean(p[:-1], q[:-1]) < 2 ** i - gamma:  # Exclude labels when calculating distance
                        idx_to_remove.add(k)

    idx_to_keep = np.array(list(set(range(s_gamma.shape[0])) - idx_to_remove))
    s_gamma = s_gamma[idx_to_keep]

    clf.fit(s_gamma[:, :-1], s_gamma[:, -1])

    pred = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    
    Accuracy_method_heuristic.append(acc)
    Compression_method_heuristic.append(s_gamma.shape[0] / S_train.shape[0])

print('Heuristic Error: ', 1 - np.mean(Accuracy_method_heuristic))
print('1/Compression Rate: ', np.mean(Compression_method_heuristic))
