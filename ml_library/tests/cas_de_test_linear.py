import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# load la lib Rust compilée
crate_root = os.path.dirname(os.path.dirname(__file__))  
if sys.platform.startswith("win"):
    lib_name = "ml_library.dll"
elif sys.platform == "darwin":
    lib_name = "libml_library.dylib"
else:
    lib_name = "libml_library.so"

lib_path = os.path.join(crate_root, "target", "debug", lib_name)
lib = ctypes.CDLL(lib_path)

# Signatures ctypes
lib.train_perceptron_binary.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_double, ctypes.c_size_t,
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]
lib.train_perceptron_binary.restype = None

lib.train_perceptron_multiclass.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_double, ctypes.c_size_t,
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]
lib.train_perceptron_multiclass.restype = None

lib.train_regression.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t,
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]
lib.train_regression.restype = None

# Wrappers Python
def train_perceptron_binary(X, Y, lr=0.01, n_iters=1000):
    n, d = X.shape
    W = np.zeros(d+1, dtype=np.float64)
    lib.train_perceptron_binary(
        X.astype(np.float64).ravel(),
        Y.astype(np.float64),
        n, d, lr, n_iters,
        W
    )
    return W

def train_perceptron_multiclass(X, Y_onehot, lr=0.01, n_iters=1000):
    n, d = X.shape
    k = Y_onehot.shape[1]
    W = np.zeros((k, d+1), dtype=np.float64)
    lib.train_perceptron_multiclass(
        X.astype(np.float64).ravel(),
        Y_onehot.astype(np.float64).ravel(),
        n, d, k, lr, n_iters,
        W.ravel()
    )
    return W

def train_regression(X, Y):
    n, d = X.shape
    W = np.zeros(d+1, dtype=np.float64)
    lib.train_regression(
        X.astype(np.float64).ravel(),
        Y.astype(np.float64),
        n, d,
        W
    )
    return W

# Fonctions de tracé
def plot_classification(X, Y, train_fn, multiclass=False, title=""):
    plt.figure()

    # 1) Fond : grille centrée autour des nuages de points
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    grid = np.array([
        [x, y]
        for x in np.linspace(x_min, x_max, 200)
        for y in np.linspace(y_min, y_max, 200)
    ])

    if multiclass:
        W = train_fn(X, Y)
        Xg_aug = np.hstack([np.ones((len(grid),1)), grid])
        scores = Xg_aug @ W.T
        preds = np.argmax(scores, axis=1)
        bg_colors = ['lightblue' if p==0 else 'pink' if p==1 else 'lightgreen'
                     for p in preds]
    else:
        W = train_fn(X, Y)
        Xg_aug = np.hstack([np.ones((len(grid),1)), grid])
        preds = np.sign(Xg_aug @ W)
        bg_colors = ['lightblue' if p>=0 else 'pink' for p in preds]

    plt.scatter(
        grid[:,0], grid[:,1],
        c=bg_colors,
        s=20, alpha=0.2,
        marker='.', zorder=0
    )

    # 2) Points originaux par dessus
    if multiclass:
        for idx, color, label in zip(range(Y.shape[1]),
                                     ['blue','red','green'],
                                     ['class0','class1','class2']):
            pts = X[np.array([y[idx] for y in Y])==1]
            plt.scatter(
                pts[:,0], pts[:,1],
                c=color, label=label,
                s=50, zorder=1
            )
    else:
        plt.scatter(
            X[Y==1,0], X[Y==1,1],
            c='blue', label='+1',
            s=50, zorder=1
        )
        plt.scatter(
            X[Y==-1,0], X[Y==-1,1],
            c='red', label='-1',
            s=50, zorder=1
        )

    plt.title(title)
    plt.legend()
    plt.show()


def plot_regression(X, Y, title="", is3d=False):
    fig = plt.figure()
    if is3d:
        ax = fig.add_subplot(111, projection='3d')
        # 1) Surface en fond
        grid = np.array([[x,y] for x in np.linspace(np.min(X[:,0]), np.max(X[:,0]), 30)
                                 for y in np.linspace(np.min(X[:,1]), np.max(X[:,1]), 30)])
        W = train_regression(X, Y)
        Xg_aug = np.hstack([np.ones((len(grid),1)), grid])
        Z = Xg_aug @ W
        ax.plot_trisurf(grid[:,0], grid[:,1], Z,
                        alpha=0.3, zorder=0)
        # 2) Points par‑dessus
        ax.scatter(X[:,0], X[:,1], Y,
                   c='black', s=30, depthshade=False, zorder=1)
    else:
        # 1) Ligne de régression
        xs = np.linspace(np.min(X), np.max(X), 100)
        W = train_regression(X, Y)
        Xs_aug = np.vstack([np.ones_like(xs), xs]).T
        ys = Xs_aug @ W
        plt.plot(xs, ys, 'r-', zorder=0)
        # 2) Points par‑dessus
        plt.scatter(X, Y, c='black', s=30, zorder=1)

    plt.title(title)
    plt.show()

# Batteries de tests 
# 1. Classification – Linear Simple
X1 = np.array([[1,1], [2,3], [3,3]])
Y1 = np.array([1, -1, -1])
plot_classification(X1, Y1, train_perceptron_binary,
                    multiclass=False, title="1. Classification – Linear Simple")

# 2. Classification – Linear Multiple
np.random.seed(0)
X2 = np.concatenate([
    np.random.random((50,2))*0.9 + np.array([1,1]),
    np.random.random((50,2))*0.9 + np.array([2,2])
])
Y2 = np.concatenate([np.ones(50), -np.ones(50)])
plot_classification(X2, Y2, train_perceptron_binary,
                    multiclass=False, title="2. Classification – Linear Multiple")

# 3. Classification – XOR
X3 = np.array([[1,0], [0,1], [0,0], [1,1]])
Y3 = np.array([1, 1, -1, -1])
plot_classification(X3, Y3, train_perceptron_binary,
                    multiclass=False, title="3. Classification – XOR")

# 4. Classification – Cross
X4 = np.random.uniform(-1,1,(500,2))
Y4 = np.where((np.abs(X4[:,0])<=0.3)|(np.abs(X4[:,1])<=0.3), 1, -1)
plot_classification(X4, Y4, train_perceptron_binary,
                    multiclass=False, title="4. Classification – Cross")

# 5. Classification – Multi Linear 3 classes
X5 = np.random.uniform(-1,1,(500,2))

def label5(p):
    x, y = p
    if -x - y - 0.5 > 0 and y < 0 and x - y - 0.5 < 0:
        return [1,0,0]
    if -x - y - 0.5 < 0 and y > 0 and x - y - 0.5 < 0:
        return [0,1,0]
    if -x - y - 0.5 < 0 and y < 0 and x - y - 0.5 > 0:
        return [0,0,1]
    return None  # points hors des trois régions

X5_list = []
Y5_list = []
for p in X5:
    lab = label5(p)
    if lab is not None:
        X5_list.append(p)
        Y5_list.append(lab)

X5 = np.array(X5_list)  
Y5 = np.array(Y5_list)   

plot_classification(
    X5, Y5, train_perceptron_multiclass, True,
    "5. Classification – Multi Linear 3 classes"
)

# 6. Classification – Multi Cross
X6 = np.random.uniform(-1,1,(1000,2))
def label6(p):
    x,y = p
    if abs(x%0.5)<=0.25 and abs(y%0.5)>0.25: return [1,0,0]
    if abs(x%0.5)>0.25 and abs(y%0.5)<=0.25: return [0,1,0]
    return [0,0,1]
Y6 = np.array([label6(p) for p in X6])
plot_classification(X6, Y6, train_perceptron_multiclass,
                    multiclass=True, title="6. Classification – Multi Cross")

# 7. Régression – Linear Simple 2D
X7 = np.array([[1],[2]])
Y7 = np.array([2,3])
plot_regression(X7, Y7, title="7. Régression – Linear Simple 2D", is3d=False)

# 8. Régression – Non Linéaire Simple 2D
X8 = np.array([[1],[2],[3]])
Y8 = np.array([2,3,2.5])
plot_regression(X8, Y8, title="8. Régression – Non Linéaire Simple 2D", is3d=False)

# 9. Régression – Linear Simple 3D
X9 = np.array([[1,1],[2,2],[3,1]])
Y9 = np.array([2,3,2.5])
plot_regression(X9, Y9, title="9. Régression – Linear Simple 3D", is3d=True)

# 10. Régression – Linear Tricky 3D
X10 = np.array([[1,1],[2,2],[3,3]])
Y10 = np.array([1,2,3])
plot_regression(X10, Y10, title="10. Régression – Linear Tricky 3D", is3d=True)

# 11. Régression – Non Linéaire Simple 3D
X11 = np.array([[1,0],[0,1],[1,1],[0,0]])
Y11 = np.array([2,1,-2,-1])
plot_regression(X11, Y11, title="11. Régression – Non Linéaire Simple 3D", is3d=True)
