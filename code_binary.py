import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

## ----------------
# Data Preparation 
## ----------------
data = np.loadtxt('diabetic_binary.csv', delimiter=',')
n = data.shape[0]
n_train = int(0.5*n)
n_valid = int(0.2*n)
n_test = int(0.25*n)
x_train = data[0:n_train,0:-1]
y_train = data[0:n_train,-1]
x_valid = data[n_train:n_train+n_valid,0:-1]
y_valid = data[n_train:n_train+n_valid,-1]
x_test = data[n-n_test:,0:-1]
y_test = data[n-n_test:,-1]

## ----------------
# Hyper-Parameters
## ----------------
sample_ratios = [0.0005, 0.001, 0.005, 0.01, 0.02]
size_subgroup = 8 
num_class = 2
k = 3
iterations = 10
lam = 0.001

## ------------------------
# Method 1: standard k-NN
## ------------------------
model = KNeighborsClassifier(n_neighbors=k)
print("Run kNN")
model.fit(x_train, y_train)
y_pred_prob = model.predict_proba(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
acc_knn = accuracy_score(y_test, y_pred)
y_valid_knn = model.predict_proba(x_valid)

ACC_knn = []
ACC_sub = []
ACC_bag = []
ACC_rse = []
ACC_urse = []

for sample_ratio in sample_ratios:
    
    ## -----------------
    # Configure Subsets 
    ## -----------------
    size_compressed = int(sample_ratio * n_train)
    num_subgroup = int(size_compressed/size_subgroup)
    
    acc_sub = []
    acc_bag = []
    acc_rse = []
    acc_urse = []
    
    for iteration in range(iterations):
        
        print("1/Compression Rate:", sample_ratio)
        print("Iteration: ", iteration)
        
        ## ---------------------
        # Method 2: SubSampling
        ## ---------------------
        print("Run Subsampling")
        idx_subtrain = np.random.choice(n_train, size=size_compressed, replace=True)
    	# This loop makes sure the subset contains data from all classes, 
        # otherwise its kNN inference is constant. Apply to all methods. 
        while len(np.unique(y_train[idx_subtrain]))<2:
            idx_subtrain = np.random.choice(n_train, size=size_compressed, replace=True)
        model.fit(x_train[idx_subtrain,:], y_train[idx_subtrain])
        y_pred_prob = model.predict_proba(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc_sub.append(accuracy_score(y_test, y_pred))
        
        ## -----------------
        # Method 3: Bagging
        ## -----------------
        print("Run Bagging")
        y_pred_group = []
        for i in range(num_subgroup):
            idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            while len(np.unique(y_train[idx_subtrain]))<2:
                idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            model.fit(x_train[idx_subtrain], y_train[idx_subtrain])
            y_pred_prob = model.predict_proba(x_test)
            y_pred = y_pred_prob
            if i==0:
                y_pred_group = np.argmax(y_pred_prob, axis=1).reshape(-1,1)
            else:
                y_pred_group = np.concatenate([y_pred_group, np.argmax(y_pred_prob, axis=1).reshape(-1,1)], axis=1)
        y_pred = []
        for i in range(n_test):
            y_pred.append(np.argmax(np.bincount(y_pred_group[i,:])))
        acc_bag.append(accuracy_score(y_test, y_pred))
        
        ## -----------------
        # Method 4: RSE-kNN
        ## -----------------
        print("Run RSE-kNN")
        y_pred_group_valid = []
        y_pred_group_test = []
        for i in range(num_subgroup):
            idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            while len(np.unique(y_train[idx_subtrain]))<2:
                idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            model.fit(x_train[idx_subtrain], y_train[idx_subtrain])
            if i == 0:
                y_pred_group_valid = model.predict_proba(x_valid)  # Use full probability array
                y_pred_group_test = model.predict_proba(x_test)  # For multi-class, we use all class probabilities
            else:
                y_pred_group_valid = np.concatenate([y_pred_group_valid, model.predict_proba(x_valid)], axis=1)
                y_pred_group_test = np.concatenate([y_pred_group_test, model.predict_proba(x_test)], axis=1)
        y_pred_test = []        
        for c in range(num_class):
            Y_pred_valid_c = y_pred_group_valid[:, c::num_class]
            Y_pred_test_c = y_pred_group_test[:, c::num_class]
            Temp1 = np.linalg.inv(np.matmul(Y_pred_valid_c.T, Y_pred_valid_c) + lam*np.eye(num_subgroup))
            Temp2 = np.matmul(Y_pred_valid_c.T, (y_valid == c).astype(int))
            alpha = np.matmul(Temp1, Temp2)
            if c == 0:
                y_pred_test = np.matmul(Y_pred_test_c, alpha).reshape(-1, 1)
            else:
                y_pred_test = np.concatenate([y_pred_test, np.matmul(Y_pred_test_c, alpha).reshape(-1, 1)],axis=1)
        y_pred = np.argmax(y_pred_test, axis=1)
        acc_rse.append(accuracy_score(y_test, y_pred))
        
        ## ------------------
        # Method 5: uRSE-kNN
        ## ------------------
        print("Run uRSE-kNN")
        y_pred_group_valid = []
        y_pred_group_test = []
        for i in range(num_subgroup):
            idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            while len(np.unique(y_train[idx_subtrain]))<2:
                idx_subtrain = np.random.choice(len(x_train), size=size_subgroup, replace=True)
            model.fit(x_train[idx_subtrain], y_train[idx_subtrain])
            if i == 0:
                y_pred_group_valid = model.predict_proba(x_valid)  # Use full probability array
                y_pred_group_test = model.predict_proba(x_test)  # For multi-class, we use all class probabilities
            else:
                y_pred_group_valid = np.concatenate([y_pred_group_valid, model.predict_proba(x_valid)], axis=1)
                y_pred_group_test = np.concatenate([y_pred_group_test, model.predict_proba(x_test)], axis=1)
        y_pred_test = []        
        for c in range(num_class):
            Y_pred_valid_c = y_pred_group_valid[:, c::num_class]
            Y_pred_test_c = y_pred_group_test[:, c::num_class]
            Temp1 = np.linalg.inv(np.matmul(Y_pred_valid_c.T, Y_pred_valid_c) + lam*np.eye(num_subgroup))
            Temp2 = np.matmul(Y_pred_valid_c.T, y_valid_knn[:,c])  
            alpha = np.matmul(Temp1, Temp2)
            if c == 0:
                y_pred_test = np.matmul(Y_pred_test_c, alpha).reshape(-1, 1)
            else:
                y_pred_test = np.concatenate([y_pred_test, np.matmul(Y_pred_test_c, alpha).reshape(-1, 1)],axis=1)
        y_pred = np.argmax(y_pred_test, axis=1)
        acc_urse.append(accuracy_score(y_test, y_pred))
      
    ACC_knn.append(acc_knn)
    ACC_sub.append(np.mean(acc_sub))
    ACC_bag.append(np.mean(acc_bag))
    ACC_rse.append(np.mean(acc_rse))
    ACC_urse.append(np.mean(acc_urse))


## ------------
# Plot Figures 
## ------------
plt.figure()
plt.plot(1-np.array(ACC_knn), label='kNN', marker='o', markersize=12, fillstyle = 'none',linewidth=2, linestyle='--')
plt.plot(1-np.array(ACC_sub), label='Sub', marker='o', markersize=12, fillstyle = 'none', linewidth=2, linestyle='-')
plt.plot(1-np.array(ACC_bag), label='Bag', marker='s', markersize = 12, fillstyle = 'none', linewidth=2, linestyle='-')
plt.plot(1-np.array(ACC_rse), label='RSE', marker='v', markersize=12, fillstyle = 'none', linewidth=2, linestyle='-')
plt.plot(1-np.array(ACC_urse), label='uRSE', marker='d', markersize=12, fillstyle = 'none', linewidth=2, linestyle='-')
plt.xlabel('1 / Compression Rate', fontsize=12)
plt.xticks(range(len(sample_ratios)),sample_ratios, fontsize=12)
plt.ylabel('Classification Error', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc = 'lower left')
plt.grid(False)
plt.show()
