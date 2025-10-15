import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import time


def Image_To_Matrix(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img.resize((128,128)), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))   # (H,W,C)->(C,H,W)
    return arr

def one_hot(indices, num_classes):
    out = np.zeros((indices.size, num_classes), dtype=np.float32)
    out[np.arange(indices.size), indices] = 1.0
    return out


Path = "PlantVillagePhotos"
rows = []
for label in os.listdir(Path):
    label_dir = os.path.join(Path, label)
    if not os.path.isdir(label_dir):
        continue
    for filename in os.listdir(label_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            rows.append({"filepath": os.path.join(label_dir, filename), "label": label})

df = pd.DataFrame(rows)
print("Total images:", len(df), "classes:", df['label'].nunique())


samples = []
labels = []
for i, row in df.iterrows():
    try:
        samples.append(Image_To_Matrix(row['filepath']))
        labels.append(row['label'])
    except Exception as e:
        print("Failed:", row['filepath'], "|", e)

X = np.stack(samples, axis=0).astype(np.float32)  # (N, C, H, W)
unique_labels = sorted(df['label'].unique())
label2idx = {lab: i for i, lab in enumerate(unique_labels)}
y_idx = np.array([label2idx[l] for l in labels], dtype=np.int32)
num_classes = len(unique_labels)

print("Loaded X:", X.shape, "num_classes:", num_classes)


# Conv1 
F1 = 8
K1 = 3


# Conv2 (new)
F2 = 16
K2 = 3



FC_hidden = 64  


lr = 1e-3


def he_init(shape, fan_in):
    return (np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in))


C = 3
kernels1 = he_init((F1, C, K1, K1), C * K1 * K1)
biases1  = np.zeros(F1, dtype=np.float32)


kernels2 = he_init((F2, F1, K2, K2), F1 * K2 * K2)
biases2  = np.zeros(F2, dtype=np.float32)


print("Initialized conv kernels shapes:", kernels1.shape, kernels2.shape)


def conv_forward_single(img, kernels, biases):
    # img: (C_in, H, W)
    # kernels: (F, C_in, K, K)
    C_in, H, W = img.shape
    F, _, K, _ = kernels.shape
    H_out = H - K + 1
    W_out = W - K + 1
    Y = np.zeros((F, H_out, W_out), dtype=np.float32)
    for f in range(F):
        Kf = kernels[f]
        bf = biases[f]
        for i in range(H_out):
            for j in range(W_out):
                s = 0.0
                # accumulate convolution
                for c in range(C_in):
                    for u in range(K):
                        for v in range(K):
                            s += Kf[c, u, v] * img[c, i+u, j+v]
                s += bf
                Y[f, i, j] = s
    return Y

def conv_backward_single(dY, img, kernels):
    # returns dK (F,C,K,K), db (F,), dX (C,H,W)
    F, C_in, K, _ = kernels.shape
    C_in_img, H, W = img.shape
    H_out, W_out = dY.shape[1], dY.shape[2]  # dY shape = (F, H_out, W_out)
    dK = np.zeros_like(kernels, dtype=np.float32)
    db = np.zeros((F,), dtype=np.float32)
    dX = np.zeros_like(img, dtype=np.float32)
    # db
    for f in range(F):
        db[f] = np.sum(dY[f])
    # dK
    for f in range(F):
        for c in range(C_in):
            for u in range(K):
                for v in range(K):
                    acc = 0.0
                    for i in range(H_out):
                        for j in range(W_out):
                            acc += img[c, i+u, j+v] * dY[f, i, j]
                    dK[f, c, u, v] = acc
    # dX (full-conv of dY with rotated kernels)
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                acc = 0.0
                for f in range(F):
                    Kf = kernels[f]
                    for u in range(K):
                        for v in range(K):
                            out_i = i - u
                            out_j = j - v
                            if 0 <= out_i < H_out and 0 <= out_j < W_out:
                                acc += Kf[c, u, v] * dY[f, out_i, out_j]
                dX[c, i, j] = acc
    return dK, db, dX

def relu_forward(X):
    return np.maximum(0.0, X)

def relu_backward(dA, A):
    
    mask = (A > 0).astype(np.float32)
    return dA * mask

def maxpool_forward_single(A, pool_h=2, pool_w=2):
    # A: (C, H, W) -> returns P: (C, H//2, W//2) and mask to use in backward
    C, H, W = A.shape
    out_h = H // pool_h
    out_w = W // pool_w
    P = np.zeros((C, out_h, out_w), dtype=np.float32)
    mask = np.zeros_like(A, dtype=np.float32)  # store mask of max positions (1.0 at max)
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                hi = i * pool_h
                wj = j * pool_w
                window = A[c, hi:hi+pool_h, wj:wj+pool_w]
                max_val = np.max(window)
                P[c, i, j] = max_val
                # set mask
                # if multiple maxima, this marks all max positions (ok for backward here)
                mask[c, hi:hi+pool_h, wj:wj+pool_w] = (window == max_val).astype(np.float32)
    return P, mask

def maxpool_backward_single(dP, mask, pool_h=2, pool_w=2):

    C = dP.shape[0]
    out_h = dP.shape[1]
    out_w = dP.shape[2]

    # dA must match the original activation spatial dimensions (mask)
    H = mask.shape[1]
    W = mask.shape[2]
    dA = np.zeros_like(mask, dtype=np.float32)

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                hi = i * pool_h
                wj = j * pool_w
                # slice of the mask corresponding to this pooling window
                window_mask = mask[c, hi:hi+pool_h, wj:wj+pool_w]
                count = np.sum(window_mask)
                if count == 0:
                    # no max marked (shouldn't normally happen), skip
                    continue
                # distribute gradient equally among the max positions
                dA[c, hi:hi+pool_h, wj:wj+pool_w] += (dP[c, i, j] / count) * window_mask

    return dA



def model_forward_single(img, params):
    # params contains kernels1, biases1, kernels2, biases2, dense weights
    kernels1, biases1, kernels2, biases2, W1, b1, W2, b2 = params
    # Conv1
    Z1 = conv_forward_single(img, kernels1, biases1)  
    A1 = relu_forward(Z1)
    P1, mask1 = maxpool_forward_single(A1, 2, 2)     
    # Conv2: input is P1 (F1 channels)
    Z2 = conv_forward_single(P1, kernels2, biases2)   
    A2 = relu_forward(Z2)
    P2, mask2 = maxpool_forward_single(A2, 2, 2)      
    # Flatten
    flat = P2.reshape(-1)                             
    # Dense1
    h = np.dot(W1, flat) + b1                         
    hA = np.maximum(0.0, h)
    # Dense2 
    logits = np.dot(W2, hA) + b2                    
    # softmax probabilities 
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    cache = (Z1, A1, P1, mask1, Z2, A2, P2, mask2, flat, h, hA, probs)
    return probs, cache

img0 = X[0]
Z1_0 = conv_forward_single(img0, kernels1, biases1)
A1_0 = relu_forward(Z1_0)
P1_0, mask1_0 = maxpool_forward_single(A1_0, 2, 2)
Z2_0 = conv_forward_single(P1_0, kernels2, biases2)
A2_0 = relu_forward(Z2_0)
P2_0, mask2_0 = maxpool_forward_single(A2_0, 2, 2)
flatten_size = P2_0.size
print("Flatten size after Conv2+Pool:", flatten_size)

# Initialize dense weights
W1 = np.random.randn(FC_hidden, flatten_size).astype(np.float32) * 0.01
b1 = np.zeros(FC_hidden, dtype=np.float32)
W2 = np.random.randn(num_classes, FC_hidden).astype(np.float32) * 0.01
b2 = np.zeros(num_classes, dtype=np.float32)

# Pack params
params = (kernels1, biases1, kernels2, biases2, W1, b1, W2, b2)


def softmax_cross_entropy_loss_and_grad(logits, y_true_onehot):
    # logits: (C,) unnormalized
    expv = np.exp(logits - np.max(logits))
    probs = expv / np.sum(expv)
    loss = -np.sum(y_true_onehot * np.log(probs + 1e-12))
    # gradient wrt logits: probs - y_true
    grad_logits = probs - y_true_onehot
    return loss, grad_logits, probs


def model_backward_single(dloss_logits, cache, params):
    # dloss_logits: gradient of loss wrt logits (num_classes,)
    kernels1, biases1, kernels2, biases2, W1, b1, W2, b2 = params
    Z1, A1, P1, mask1, Z2, A2, P2, mask2, flat, h, hA, probs = cache
    # Dense2 grad
    dW2 = np.outer(dloss_logits, hA)           # (num_classes, FC_hidden)
    db2 = dloss_logits.copy()
    # backprop into hidden activation hA
    dhA = np.dot(W2.T, dloss_logits)           # (FC_hidden,)
    # ReLU backward for hidden h
    dh = dhA * (h > 0).astype(np.float32)      # (FC_hidden,)
    dW1 = np.outer(dh, flat)                   # (FC_hidden, flatten_size)
    db1 = dh.copy()
    # backprop to flattened conv output
    dflat = np.dot(W1.T, dh)                   # (flatten_size,)
    # reshape to P2 shape
    dP2 = dflat.reshape(P2.shape)              # (F2, H2p, W2p)
    # pool2 backward -> dA2
    dA2 = maxpool_backward_single(dP2, mask2, 2, 2)
    # relu2 backward: need pre-activation Z2 to mask
    dZ2 = dA2 * (Z2 > 0).astype(np.float32)
    # conv2 backward: dK2, db2_conv, dP1 (gradient into pooled output from conv1)
    dK2, db2_conv, dP1 = conv_backward_single(dZ2, P1, kernels2)
    # pool1 backward -> dA1
    dA1 = maxpool_backward_single(dP1, mask1, 2, 2)
    # relu1 backward
    dZ1 = dA1 * (Z1 > 0).astype(np.float32)
    # conv1 backward to get gradients and dX if needed
    dK1, db1_conv, dX = conv_backward_single(dZ1, img0, kernels1)
    # Package gradients
    grads = {
        "dK1": dK1, "db1": db1_conv,
        "dK2": dK2, "db2_conv": db2_conv,
        "dW1": dW1, "db1_dense": db1,
        "dW2": dW2, "db2_dense": db2
    }
    return grads



n_train = 32 #X.shape[0] to train the model for all the images
idxs = np.random.choice(X.shape[0], n_train, replace=False)
X_train = X[idxs]
y_train = y_idx[idxs]
y_onehot = one_hot(y_train, num_classes)

epochs = 10 #maybe less epochs if trianing entire thing
print("Training on tiny subset of size", n_train, "for", epochs, "epochs .")

for epoch in range(epochs):
    t0 = time.time()
    total_loss = 0.0
    # simple SGD over the subset, sample order shuffled
    order = np.random.permutation(n_train)
    for ii in order:
        x_sample = X_train[ii]
        y_true = y_onehot[ii]
        # forward
        probs, cache = model_forward_single(x_sample, params)
        loss, dlogits, _ = softmax_cross_entropy_loss_and_grad(np.log(probs + 1e-12), y_true)
        
        total_loss += loss
        # backward
        grads = model_backward_single(dlogits, cache, params)
        # parameter updates (SGD)
        # conv1
        kernels1 -= lr * grads["dK1"]
        biases1  -= lr * grads["db1"]
        # conv2
        kernels2 -= lr * grads["dK2"]
        biases2  -= lr * grads["db2_conv"]
        # dense
        W1 -= lr * grads["dW1"]
        b1 -= lr * grads["db1_dense"]
        W2 -= lr * grads["dW2"]
        b2 -= lr * grads["db2_dense"]
        # re-pack params for forward
        params = (kernels1, biases1, kernels2, biases2, W1, b1, W2, b2)
    t1 = time.time()
    print("Epoch", epoch+1, "loss (avg):", total_loss / n_train, "time:", round(t1-t0,2), "s")

print("Training done. Final params shapes:")
print("kernels1", kernels1.shape, "kernels2", kernels2.shape, "W1", W1.shape, "W2", W2.shape)

# End of script
