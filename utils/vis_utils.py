import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation
from IPython.display import HTML
import matplotlib.image as mpimg 
from math import atan, cos, sin,acos,pi,sqrt

import numpy as np
import matplotlib.lines as mlines
import os 

def visualize_db(classifier, X=None,Y=None,x_min=None,x_max=None,y_min=None,y_max=None):
    if(X is not None and Y is not None):
        x_min,x_max = X[:,0].min(), X[:,0].max()
        y_min,y_max = X[:,1].min(), X[:,1].max()
    x_min = x_min *1.2
    y_min = y_min *1.2
    x_max = x_max*1.2
    y_max = y_max*1.2
    
    xx, yy = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = classifier.predict_proba(grid)
    probs = probs[:,1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contour(xx, yy, probs, 1, cmap="Greys", vmin=0.0, vmax=0.9)
    sns.scatterplot(x =X[:,0],y=X[:,1],hue=(Y+1)/2,ax=ax)
    return ax


class UnitCircleCanvas:
    
    def __init__(self,r,ax):
        ax = plt.gca()
        ax.cla() 
        circle1 = plt.Circle((0, 0), r, color='b', fill=False)
        circle2 = plt.Circle((0, 0), 0.025, color='black', fill=False)
        ax.set_xlim((-r-0.2, r+0.2))
        ax.set_ylim((-r-0.2, r+0.2))
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        self.r = r
        self.ax = ax
    
    def unit_vector_at_angle(self,w,theta):
        r1 = 1
        theta_0 = atan(w[1]/w[0])
        theta_1 = theta_0 + theta
        w2 = np.array([ r1* cos(theta_1) , r1*sin(theta_1)])
        return w2
    
    def draw_line(self,w,b,color='b',linestyle='-',lw=2.,label=None):
        x1,x2 = self.line_end_points_on_circle_2d(w[0],w[1],b)
        #print(x1,x2)
        x_arr = np.arange(x1-1e-5,x2+1e-5,1e-5)
        m     = w[0]/w[1]
        y1    = -m*x_arr - b/w[1] 
        if(label is None):
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle)
        else:
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle,label=label)
        self.ax.add_line(line1)
        
    
    def draw_line_at_angle(self,w,b,theta,color='b',linestyle='-',lw=2.,label=None):
        w2 = self.unit_vector_at_angle(w,theta)
        #print(w2)
        self.draw_line(w2,b,color=color,linestyle=linestyle,lw=lw,label=label)
    
    def line_end_points_on_circle_2d(self,w0,w1,b):
        # w_0x + w_1y +b =0
        # x^2 + y^2 = r^2
        m1 = w0/w1
        m2 = b/w1
        a = 1 + m1**2
        b = 2*m1*m2
        c = m2**2 - self.r**2
        d = b**2 - 4*a*c
        if(d<1e-6):
            d = 0
        d = sqrt(d)
        x1 = (-b-d)/(2*a)
        x2 = (-b+d)/(2*a)
        return x1,x2
    

class RectangleCanvas:
    
    def __init__(self,x_min,x_max,y_min,y_max,ax):
        ax = plt.gca()
        ax.cla() 
        
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        #ax.add_patch(circle1)
        #ax.add_patch(circle2)
        
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min

        self.ax = ax
    
    def unit_vector_at_angle(self,w,theta):
        r1 = 1
        theta_0 = atan(w[1]/w[0])
        theta_1 = theta_0 + theta
        w2 = np.array([ r1* cos(theta_1) , r1*sin(theta_1)])
        return w2
    
    def draw_line(self,w,b,color='b',linestyle='-',lw=2.,label=None):
        #x1,x2 = self.line_end_points_on_circle_2d(w[0],w[1],b)
        #print(x1,x2)
        x_arr = np.arange(self.x_min, self.x_max, 1e-5)
        m     = w[0]/w[1]
        y1    = -m*x_arr - b/w[1] 
        if(label is None):
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle)
        else:
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle,label=label)
        self.ax.add_line(line1)
        
    
    def draw_line_at_angle(self,w,b,theta,color='b',linestyle='-',lw=2.,label=None):
        w2 = self.unit_vector_at_angle(w,theta)
        #print(w2)
        self.draw_line(w2,b,color=color,linestyle=linestyle,lw=lw,label=label)
    

def split_pos_neg_pts(lst_idx,Y):
    qp = []
    qn = []
    for i in lst_idx:
        if(Y[i]==1):
            qp.append(i)
        else:
            qn.append(i)
    return qp,qn
 
def vis_dataset(dataset):
    Y = dataset.Y 
    X = dataset.X 
    idx_p = np.where(Y==1)[0]
    idx_n = np.where(Y==0)[0]
    plt.figure(figsize=(6,6))
    plt.scatter(X[idx_p,0],X[idx_p,1],s=1)
    plt.scatter(X[idx_n,0],X[idx_n,1],s=1)