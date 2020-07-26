import numpy as np
from sklearn import decomposition

scoreData = np.mat([
[5,2,1,4,0,0,2,4,0,0,0],
[0,0,0,0,0,0,0,0,0,3,0],
[1,0,5,2,0,0,3,0,3,0,1],
[0,5,0,0,4,0,1,0,0,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[0,0,1,0,0,0,1,0,0,5,0],
[5,0,2,4,2,1,0,3,0,1,0],
[0,4,0,0,5,4,0,0,0,0,5],
[0,0,0,0,0,0,4,0,4,5,0],
[0,0,0,4,0,0,1,5,0,0,0],
[0,0,0,0,4,5,0,0,0,0,3],
[4,2,1,4,0,0,2,4,0,0,0],
[0,1,4,1,2,1,5,0,5,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[2,5,0,0,4,0,0,0,0,0,0],
[5,0,0,0,0,0,0,4,2,0,0],
[0,2,4,0,4,3,4,0,0,0,0],
[0,3,5,1,0,0,4,1,0,0,0]
])

def cosSim(vec_1, vec_2):
    dotProd = float(np.dot(vec_1.T, vec_2))
    normProd = np.linalg.norm(vec_1)*np.linalg.norm(vec_2)
    return 0.5+0.5*(dotProd/normProd)

def estScore(scoreData,scoreData_PCA,userIndex,itemIndex):
    n = np.shape(scoreData)[1]
    simSum = 0
    simSumScore = 0
    for i in range(n):
        userScore = scoreData[userIndex,i]
        if userScore == 0 or i == itemIndex:
            continue
        sim = cosSim(scoreData_PCA[:, i], scoreData_PCA[:, itemIndex])
        simSum = float(simSum + sim)
        simSumScore = simSumScore + userScore * sim
    if simSum == 0:
        return 0
    return simSumScore / simSum


scoreData_mean = scoreData - np.mean(scoreData, 0)
cov_mat = np.cov(scoreData_mean, rowvar=False)
eignvalue, featurevector = np.linalg.eig(cov_mat)


sigmaSum = 0
k_num = 0

for k in range(len(eignvalue)):
    sigmaSum = sigmaSum + eignvalue[k]
    if float(sigmaSum)/float(np.sum(eignvalue)) > 0.9:
        k_num = k+1
        break

pca = decomposition.PCA(n_components=k_num)
scoreData_PCA = pca.fit_transform(scoreData_mean).T


n = np.shape(scoreData)[1]
userIndex = 18
for i in range(userIndex):
    for j in range(n):
        userScore = scoreData[i, j]
        if userScore != 0:
            continue
        print("user:{},index:{},score:{}".format(i, j, estScore(scoreData, scoreData_PCA, i, j)))