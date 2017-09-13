# Austin Hester
# 09/13/2017
# PLA Python Implementation
# Trains with 50 linearly seperable points
# Tests against 30
import numpy as np
import random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, N):
        x1, y1, x2, y2 = [random.uniform(-1, 1) for i in range(4)]
        # for generating linearly seperable data (V)
        self.V = np.array([x2*y1-x1*y2, y2-y1, x1-x2])
        self.X = self.generatePoints(N)

    def generatePoints(self, N):
        X = []
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            x_ = np.array([1, x1, x2])
            s = int(np.sign(self.V.T.dot(x_)))
            x_ = np.append(x_, [s])
            X.append(x_)
        return np.array(X)

    def plot(self, vec=None, save=False):
        fig = plt.figure(figsize=(6,6))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        ax = fig.add_subplot(1,1,1)
        ax.scatter(self.X[:,1:2], self.X[:,2:3], c=self.X[:,3:4], cmap='prism')
        if (vec is not None and vec[2] != 0):
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            plt.savefig('.\gifs\p_N%s' % (str(len(self.X))), dpi=100, bbox_inches='tight')
        else:
            plt.show()

    def classifyError(self, w_, pts=None):
        if pts is None:
            pts = self.X[:,:3]
            S = self.X[:,3:4]
        else:
            S = pts[:,3:4]
            pts = pts[:,:3]
        M = len(pts)
        n_mispts = 0
        for x_, s in zip(pts, S):
            print(x_, ": ", s)
            if int(np.sign(w_.T.dot(x_))) != s:
                n_mispts += 1
        print(n_mispts)
        err = n_mispts / float(M)
        return err

    def pickMisclPoint(self, w_):
        pts = self.X[:,:3]
        S = self.X[:,3:4]
        mispts = []
        for x,s in zip(pts, S):
            if int(np.sign(w_.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0, len(mispts))]

    def pla(self, save=False):
        X = self.X[:,:3]
        print(self.X)
        N = len(X)
        w_ = np.zeros(len(X[0]))
        it = 0
        print(self.classifyError(w_))
        while self.classifyError(w_) != 0:
            it += 1
            # pick mispicked pt
            x, s = self.pickMisclPoint(w_)
            w_ += s*x
            if save:
                self.plot(vec=w_, save=True)
                plt.title('N = %s, Iteration %s\n' % (str(N),str(it)))
                plt.savefig('.\gifs\p_N%s_it%s' % (str(N),str(it)), dpi=100, bbox_inches='tight')
        self.w = w_

    def checkTestError(self, M, w_):
        testPts = self.generatePoints(M)
        return self.classifyError(w_, pts=testPts)
        

p = Perceptron(50)
#p.plot()
p.pla()
print(p.checkTestError(30, p.w))
p.plot(vec=p.w)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(p.X[:,1:2], p.X[:,2:3], c=p.X[:,3:4], cmap='prism')
#plt.show()
