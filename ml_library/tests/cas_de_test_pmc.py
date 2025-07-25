import os
import ctypes
import platform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# nom de la crate
CRATE_NAME = "ml_library"

# choisir l'extension dynamique
if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
elif platform.system() == "Darwin":
    lib_fname = f"lib{CRATE_NAME}.dylib"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

# chemin absolu vers target/release
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
lib_path = os.path.join(root, "target", "release", lib_fname)
print(">>> Library path:", lib_path, "exists?", os.path.exists(lib_path))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Bibliothèque introuvable : {lib_path}")

lib = ctypes.CDLL(lib_path)

# signature FFI
lib.train_pmc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # X
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Y
    ctypes.c_size_t,  # n_samples
    ctypes.c_size_t,  # n_features
    ctypes.c_size_t,  # n_hidden_layers
    np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),    # hidden_sizes
    ctypes.c_size_t,  # n_outputs
    ctypes.c_size_t,  # hidden_act (0=ReLU,1=Tanh)
    ctypes.c_size_t,  # output_act (0=Sigmoid,1=Softmax,2=Linear)
    ctypes.c_double,  # lr
    ctypes.c_size_t,  # n_iters
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out weights
]
lib.train_pmc.restype = None

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=5000):
    n, d = X.shape
    hs = np.array(hidden_sizes, dtype=np.int64)

    # préparer Y_flat et k
    if output_act == 2:
        k = 1
        Y_flat = Y.astype(np.float64).ravel()
    elif output_act == 1:
        k = Y.shape[1]
        Y_flat = Y.astype(np.float64).ravel()
    else:
        k = 1
        Yb = (Y > 0).astype(np.float64)
        Y_flat = Yb.ravel()

    layers = [d] + hidden_sizes + [k]
    total = sum((layers[i] + 1) * layers[i+1] for i in range(len(layers)-1))
    W = np.zeros(total, dtype=np.float64)

    print(f">>> [PY] train_pmc called with X.shape={X.shape}, hidden_sizes={hidden_sizes}, hidden_act={hidden_act}, output_act={output_act}, n_iters={n_iters}")

    lib.train_pmc(
        X.astype(np.float64),
        Y_flat,
        n, d, len(hidden_sizes), hs,
        k, hidden_act, output_act,
        lr, n_iters,
        W
    )
    return W, layers

def pmc_forward(X, W, layers, hidden_act, output_act):
    weights = []
    idx = 0
    for i in range(len(layers)-1):
        out_sz, in_sz = layers[i+1], layers[i] + 1
        cnt = out_sz * in_sz
        mat = W[idx:idx+cnt].reshape(out_sz, in_sz)
        weights.append(mat)
        idx += cnt

    H = X.copy()
    for i, Wm in enumerate(weights):
        H = np.hstack([np.ones((H.shape[0],1)), H])
        Z = H.dot(Wm.T)
        if i < len(weights)-1:
            H = np.maximum(0, Z) if hidden_act == 0 else np.tanh(Z)
        else:
            if output_act == 0:
                H = 1/(1+np.exp(-Z))
            elif output_act == 1:
                e = np.exp(Z - Z.max(axis=1, keepdims=True))
                H = e / e.sum(axis=1, keepdims=True)
            else:
                H = Z
    return H

def plot_classification(X, Y, hidden_sizes, hidden_act, output_act, multiclass=False, title=""):
    plt.figure()
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    mx, my = (x_max-x_min)*0.1, (y_max-y_min)*0.1
    xs = np.linspace(x_min-mx, x_max+mx, 200)
    ys = np.linspace(y_min-my, y_max+my, 200)
    grid = np.array([[x,y] for x in xs for y in ys])

    W, layers = train_pmc(X, Y, hidden_sizes, hidden_act, output_act)
    preds = pmc_forward(grid, W, layers, hidden_act, output_act)

    if multiclass:
        cl = np.argmax(preds, axis=1)
        cols = ['lightblue' if p==0 else 'pink' if p==1 else 'lightgreen' for p in cl]
    else:
        probs = preds.ravel()
        cols = ['lightblue' if p>=0.5 else 'pink' for p in probs]

    plt.scatter(grid[:,0], grid[:,1], c=cols, alpha=0.2, zorder=0)
    if multiclass:
        for c,col in enumerate(['blue','red','green']):
            pts = X[[y[c]==1 for y in Y]]
            plt.scatter(pts[:,0], pts[:,1], c=col, label=f"class{c}", zorder=1)
    else:
        plt.scatter(X[Y==1,0], X[Y==1,1], c='blue',  label='+1', zorder=1)
        plt.scatter(X[Y==-1,0], X[Y==-1,1], c='red',   label='-1', zorder=1)

    plt.title(title)
    plt.legend()
    plt.show()

def plot_regression(X, Y, hidden_sizes, hidden_act, output_act, title="", is3d=False):
    plt.figure()
    W, layers = train_pmc(X, Y, hidden_sizes, hidden_act, output_act)

    if is3d:
        ax = plt.subplot(projection='3d')
        ax.scatter(X[:,0], X[:,1], Y, zorder=1)
        grid = np.array([[x,y] for x in np.linspace(X[:,0].min(), X[:,0].max(), 30)
                                 for y in np.linspace(X[:,1].min(), X[:,1].max(), 30)])
        Z = pmc_forward(grid, W, layers, hidden_act, output_act).ravel()
        ax.plot_trisurf(grid[:,0], grid[:,1], Z, alpha=0.3, zorder=0)
    else:
        plt.scatter(X.ravel(), Y, zorder=1)
        xs = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
        ys = pmc_forward(xs, W, layers, hidden_act, output_act).ravel()
        plt.plot(xs.ravel(), ys, 'r-', zorder=0)

    plt.title(title)
    plt.show()

# --- Batterie de 11 tests ---

# 1. Linear Simple
X1 = np.array([[1,1],[2,3],[3,3]])
Y1 = np.array([1,-1,-1])
plot_classification(X1, Y1, [10,10], 0, 0, False, "1. Classification – Linear Simple")

# 2. Linear Multiple
np.random.seed(0)
X2 = np.vstack([
    np.random.random((50,2))*0.9 + [1,1],
    np.random.random((50,2))*0.9 + [2,2]
])
Y2 = np.concatenate([np.ones(50), -np.ones(50)])
plot_classification(X2, Y2, [10,10], 0, 0, False, "2. Classification – Linear Multiple")

# 3. XOR
X3 = np.array([[1,0],[0,1],[0,0],[1,1]])
Y3 = np.array([1,1,-1,-1])
plot_classification(X3, Y3, [10,10], 0, 0, False, "3. Classification – XOR")

# 4. Cross
X4 = np.random.uniform(-1,1,(500,2))
Y4 = np.where((np.abs(X4[:,0])<=0.3)|(np.abs(X4[:,1])<=0.3), 1, -1)
plot_classification(X4, Y4, [10,10], 0, 0, False, "4. Classification – Cross")

# 5. Multi Linear 3 classes
X5 = np.random.uniform(-1,1,(500,2))
def label5(p):
    x,y = p
    if -x-y-0.5>0 and y<0 and x-y-0.5<0: return [1,0,0]
    if -x-y-0.5<0 and y>0 and x-y-0.5<0: return [0,1,0]
    if -x-y-0.5<0 and y<0 and x-y-0.5>0: return [0,0,1]
    return None
labels5 = [label5(p) for p in X5]
mask5 = [lab is not None for lab in labels5]
X5, Y5 = X5[mask5], np.array([lab for lab in labels5 if lab is not None], dtype=np.float64)
plot_classification(X5, Y5, [10,10], 0, 1, True,  "5. Classification – Multi Linear 3 classes")

# 6. Multi Cross (réglage hyperparamètres)
X6 = np.random.uniform(-1,1,(1000,2))
def label6(p):
    x,y = p
    if abs(x%0.5)<=0.25 and abs(y%0.5)>0.25: return [1,0,0]
    if abs(x%0.5)>0.25 and abs(y%0.5)<=0.25: return [0,1,0]
    return [0,0,1]
Y6 = np.array([label6(p) for p in X6], dtype=np.float64)
plot_classification(
    X6, Y6,
    hidden_sizes=[20,20],
    hidden_act=1,
    output_act=1,
    multiclass=True,
    title="6. Classification – Multi Cross"
)

# 7. Régression – Linear Simple 2D
X7 = np.array([[1],[2]])
Y7 = np.array([2,3])
plot_regression(X7, Y7, [10,10], 0, 2, "7. Régression – Linear Simple 2D", False)

# 8. Régression – Non Linéaire Simple 2D
X8 = np.array([[1],[2],[3]])
Y8 = np.array([2,3,2.5])
plot_regression(X8, Y8, [10,10], 0, 2, "8. Régression – Non Linéaire Simple 2D", False)

# 9. Régression – Linear Simple 3D
X9 = np.array([[1,1],[2,2],[3,1]])
Y9 = np.array([2,3,2.5])
plot_regression(X9, Y9, [10,10], 0, 2, "9. Régression – Linear Simple 3D", True)

# 10. Régression – Linear Tricky 3D
X10 = np.array([[1,1],[2,2],[3,3]])
Y10 = np.array([1,2,3])
plot_regression(X10, Y10, [10,10], 0, 2, "10. Régression – Linear Tricky 3D", True)

# 11. Régression – Non Linéaire Simple 3D
X11 = np.array([[1,0],[0,1],[1,1],[0,0]])
Y11 = np.array([2,1,-2,-1])
plot_regression(X11, Y11, [10,10], 0, 2, "11. Régression – Non Linéaire Simple 3D", True)
