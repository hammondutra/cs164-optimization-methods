"""CS164 Session 4: Positive Definiteness — Python translation of PCW.

The source screenshot was titled 'Some Linear Algebra'. Its final question
only shows a partial student answer; the gradient identity in this script is
provided as a mathematical supplement, not claimed to be in that answer.
"""
import numpy as np
import matplotlib.pyplot as plt


def symmetric_examples():
    """Numerical versions of the visible SageMath symmetric-matrix examples."""
    A1=np.array([[1.]])
    A3=np.array([[1.,2.,3.],[2.,1.,2.],[3.,2.,1.]])
    return A1,A3


def positive_definite_examples():
    """Four matrices from the source's positive-definiteness exercise."""
    return {
        'A1':np.array([[5.,6.],[6.,7.]]),
        'A2':np.array([[-1.,-2.],[-2.,-5.]]),
        'A3':np.array([[1.,10.],[10.,100.]]),
        'A4':np.array([[1.,10.],[10.,101.]]),
    }


def classify_symmetric(A,tol=1e-10):
    eigenvalues=np.linalg.eigvalsh(A)
    if np.all(eigenvalues>tol): return 'positive definite',eigenvalues
    if np.all(eigenvalues>=-tol): return 'positive semidefinite',eigenvalues
    if np.all(eigenvalues<-tol): return 'negative definite',eigenvalues
    if np.all(eigenvalues<=tol): return 'negative semidefinite',eigenvalues
    return 'indefinite',eigenvalues


def quadratic_form(A,x):
    x=np.asarray(x,dtype=float)
    return float(x@A@x)


def bowl(x,y): return x**2+y**2
def saddle(x,y): return x**2-y**2


def level_example(x,y):
    """x.T @ [[2,1],[1,2]] @ x."""
    return 2*x**2+2*x*y+2*y**2


def plot_level_sets():
    x=np.linspace(-4,4,350); X,Y=np.meshgrid(x,x)
    for f,title,levels in [
        (bowl,'Quadratic bowl: x²+y²',[0.5,1,2,4,8,12]),
        (saddle,'Saddle: x²-y²',[-8,-4,-2,-1,0,1,2,4,8])]:
        fig,ax=plt.subplots(figsize=(6,5)); C=ax.contour(X,Y,f(X,Y),levels=levels)
        ax.clabel(C,fontsize=8); ax.set(xlabel='x',ylabel='y',title=title)
        ax.set_aspect('equal'); ax.grid(True,alpha=.25)
    q=np.linspace(-2,2,350); X,Y=np.meshgrid(q,q)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.contour(X,Y,level_example(X,Y),levels=[1],colors='C0')
    ax.axhline(0,color='gray',linewidth=.6); ax.axvline(0,color='gray',linewidth=.6)
    ax.set(xlabel='x',ylabel='y',title='xᵀ[[2,1],[1,2]]x = 1')
    ax.set_aspect('equal')
    return fig


def grad_linear(b):
    """∇(bᵀx) = b (supplement to partial source answer)."""
    return np.asarray(b,dtype=float)


def grad_quadratic(A,x):
    """∇(xᵀAx) = (A+Aᵀ)x; =2Ax if A is symmetric."""
    A=np.asarray(A,dtype=float); x=np.asarray(x,dtype=float)
    return (A+A.T)@x


def main():
    for name,A in positive_definite_examples().items():
        kind,ev=classify_symmetric(A)
        print(name,'eigenvalues:',ev,'classification:',kind)
    assert quadratic_form(positive_definite_examples()['A1'],[4,-3])==-1
    assert quadratic_form(positive_definite_examples()['A2'],[1,0])==-1
    print('Counterexample det(-I)=',np.linalg.det(-np.eye(2)),
          '; -I is negative definite despite positive determinant')
    plot_level_sets(); plt.show()

if __name__=='__main__':
    main()
