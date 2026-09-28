"""CS164 Session 3: Quadratic Forms — NumPy/Matplotlib PCW translations.

Only the questions/plots visible in the submitted 5-page screenshot are included.
The screenshot stops after Q6, despite labeling the exercise Q6 of 9.
"""
import numpy as np
import matplotlib.pyplot as plt


def f1(x,y):
    return np.sin(x)**2 + np.sin(y)**2


def q1(x,y):
    return x**2 + y**2


def f2(x,y):
    return (x+y)**2 + x**4 + y**4


def q2(x,y):
    return (x+y)**2


def f_quadratic(x,y):
    return 2*x**2 + 4*x*y + y**2


def g_nonquadratic(x,y):
    return 7 + 2*(x+y)**2 - y*np.sin(y) - x**3


def q_g(x,y):
    """Order-2 Taylor polynomial for g at (0,0): 7+2x²+4xy+y²."""
    return 7 + 2*x**2 + 4*x*y + y**2


def plot_comparison(f, q, bounds=(-1.5,1.5), title=''):
    axis=np.linspace(bounds[0],bounds[1],70)
    X,Y=np.meshgrid(axis,axis)
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,f(X,Y),color='C0',alpha=.65,linewidth=0)
    ax.plot_surface(X,Y,q(X,Y),color='C1',alpha=.5,linewidth=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='C0',alpha=.65,label='Original'),
                       Patch(color='C1',alpha=.5,label='Quadratic Taylor')])
    ax.set(xlabel='x',ylabel='y',zlabel='z',title=title)
    return fig


def main():
    plot_comparison(f1,q1,title='sin²(x)+sin²(y) and x²+y²')
    plot_comparison(f2,q2,bounds=(-1,1),title='(x+y)²+x⁴+y⁴ and (x+y)²')
    plot_comparison(f_quadratic,f_quadratic,title='Quadratic equals its own approximation')
    plot_comparison(g_nonquadratic,q_g,title='7+2(x+y)²-y sin(y)-x³ and Taylor polynomial')
    plt.show()

if __name__=='__main__':
    main()
