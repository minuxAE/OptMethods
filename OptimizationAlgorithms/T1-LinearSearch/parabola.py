"""
精确线搜：抛物线法(二次插值法)
"""
from math import sqrt, sin
import matplotlib.pyplot as plt

def qmin(phi, param, a, b, delta, eps):
    s0=a; maxj=20; maxk=30; big=1e6; err=1; k=1; Err = []; tk = []
    S=[s0]; cond=0; h=1; ds=1e-5

    if abs(s0) > 1e4:
        h = abs(s0) * 1e-4
    while k<maxk and err > eps and cond != 5:
        f1=(eval(phi,globals(),{param: s0+ds})-eval(phi,globals(),{param: s0-ds}))/(2*ds)
        if f1 > 0:
            h = -abs(h)
        
        s1=s0+h; s2=s0+2*h; bars=s0
        phi0=eval(phi,globals(),{param: s0}); phi1=eval(phi,globals(),{param: s1})
        phi2=eval(phi,globals(),{param: s2}); barphi=phi0; cond=0
        j=0;  # 确定h使得phi1<phi0且phi1<phi2
        err=s2-s0
        Err.append(err)  # 生成一个残量序列
        tk.append(k)      # 生成一个迭代次数序列   
        while(j<maxj and abs(h)>delta and cond==0):
            if (phi0<=phi1):
                s2=s1; phi2=phi1; h=0.5*h
                s1=s0+h; phi1=eval(phi,globals(),{param: s1})
            elif (phi2<phi1):
                s1=s2; phi1=phi2; h=2*h
                s2=s0+2*h; phi2=eval(phi,globals(),{param: s2})
            else:
                cond=-1
            err=abs(s2-s0); 
            j=j+1

            if(abs(h)>big or abs(s0)>big): 
                cond=5
        if(cond==5):
            bars=s1; barphi=eval(phi,globals(),{param: s1})
        else:
            d=2*(2*phi1-phi0-phi2)
            if(d<0):
                barh=h*(4*phi1-3*phi0-phi2)/d
            else:
                barh=h/3; cond=4
            bars=s0+barh; barphi=eval(phi,globals(),{param: bars})
            h=abs(h); h0=abs(barh)
            h1=abs(barh-h); h2=abs(barh-2*h)
            # 确定下一次迭代的h值
            if(h0<h): h=h0
            if(h1<h): h=h1
            if(h2<h): h=h2
            if(h==0): h=barh
            if(h<delta): cond=1
            if(abs(h)>big or abs(bars)>big): cond=5
            err=abs(phi1-barphi)
            s0=bars; k=k+1; S.append(s0)
        if(cond==2 and h<delta): cond=3
    s=s0; phis=eval(phi,globals(),{param: s})
    ds=h; dphi=err;
    return s, phis, k, ds, dphi, S,Err,tk   

def main():
    s, phis, k, ds, dphi, S,Err,tk= qmin("x**2-sin(x)", "x", 0, 1, 1e-4, 1e-6)
    print(s, k, ds, dphi)
    plt.xticks(range(1,k))
    plt.yscale('log')
    plt.xlabel('Iteration: k')
    plt.ylabel('Error')
    plt.plot(tk,Err,'b-.', lw=1)
    plt.show()

if __name__ == '__main__':
    main()