︠917e7d0d-ad9c-4ec3-9352-a885ef42c9e5r︠
%auto
# This cell automatically evaluates on startup -- or run it manually if it didn't evaluate.
# Here, it initializes the Jupyter kernel with the specified name and sets it as the default mode for this worksheet.
jupyter_kernel = jupyter("python3")  # run "jupyter?" for more information.
%default_mode jupyter_kernel
︡413c16b0-89b5-4950-865c-4b899541f389︡
︠bc38d584-d310-4421-aaf2-e5d813d4b966i︠
%md
# SymPyLab

SymPy’s documentation
- https://docs.sympy.org/latest/index.html
︡a5b1866f-1467-42cf-8cb4-89c0585c783e︡{"md": "# SymPyLab\n\nSymPy\u2019s documentation\n- https://docs.sympy.org/latest/index.html", "done": true}︡
︠80b894c1-881d-4bc4-81af-9730cb0cc7adi︠
%md
## SymPy’s polynomials 
- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials 

- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10)  = x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800

- x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800 / (x-1) = x^9 - 54 x^8 + 1266 x^7 - 16884 x^6 + 140889 x^5 - 761166 x^4 + 2655764 x^3 - 5753736 x^2 + 6999840 x - 3628800
︡c02469c4-51e9-42f7-a280-648fb60b6e65︡{"md": "## SymPy\u2019s polynomials \n- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials \n\n- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10)  = x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800\n\n- x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800 / (x-1) = x^9 - 54 x^8 + 1266 x^7 - 16884 x^6 + 140889 x^5 - 761166 x^4 + 2655764 x^3 - 5753736 x^2 + 6999840 x - 3628800", "done": true}︡
︠54c10b3c-37fc-486a-909d-02a21cfd20d6i︠
%md
<img src="https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram1.jpg" /> <img src="https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram2.jpg" />
︡05fc6ab3-b6ed-4f63-97dd-78409e0a1f7f︡{"md": "<img src=\"https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram1.jpg\" /> <img src=\"https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram2.jpg\" />", "done": true}︡
︠614a7974-3b1e-48c0-9538-9e70b5f6c789︠
from sympy import Symbol
from sympy import div

x = Symbol('x')

p = x**10 - 55*x**9  + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800

p, r = div(p,  x-1)

print(p)
print(r)

p, r = div(p,  x-2)

print(p)
print(r)

p, r = div(p,  x-3)

print(p)
print(r)

p, r = div(p,  x-4)

print(p)
print(r)

p, r = div(p,  x-5)

print(p)
print(r)

p, r = div(p,  x-6)

print(p)
print(r)

p, r = div(p,  x-7)

print(p)
print(r)

p, r = div(p,  x-8)

print(p)
print(r)

p, r = div(p,  x-9)

print(p)
print(r)
︡2abfd3a8-15a5-4bb5-b044-ce51060edbff︡{"html": "<pre><span style=\"font-family:monospace;\">x**9 - 54*x**8 + 1266*x**7 - 16884*x**6 + 140889*x**5 - 761166*x**4 + 2655764*x**3 - 5753736*x**2 + 6999840*x - 3628800\n0\nx**8 - 52*x**7 + 1162*x**6 - 14560*x**5 + 111769*x**4 - 537628*x**3 + 1580508*x**2 - 2592720*x + 1814400\n0\nx**7 - 49*x**6 + 1015*x**5 - 11515*x**4 + 77224*x**3 - 305956*x**2 + 662640*x - 604800\n0\nx**6 - 45*x**5 + 835*x**4 - 8175*x**3 + 44524*x**2 - 127860*x + 151200\n0\nx**5 - 40*x**4 + 635*x**3 - 5000*x**2 + 19524*x - 30240\n0\nx**4 - 34*x**3 + 431*x**2 - 2414*x + 5040\n0\nx**3 - 27*x**2 + 242*x - 720\n0\nx**2 - 19*x + 90\n0\nx - 10\n0\n</span></pre>", "done": true}︡
︠185c7315-d00a-404d-845d-621e479f7f0di︠
%md
## SymPy’s polynomial simple univariate polynomial factorization
- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization
- factor(x\*\*10 - 55\*x\*\*9  + 1320\*x\*\*8 - 18150\*x\*\*7 + 157773\*x\*\*6 - 902055\*x\*\*5 + 3416930\*x\*\*4 - 8409500\*x\*\*3 + 12753576\*x\*\*2 - 10628640\*x + 3628800)
︡742da4aa-90b8-4820-8407-3d2ebb2d0a29︡{"md": "## SymPy\u2019s polynomial simple univariate polynomial factorization\n- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization\n- factor(x\\*\\*10 - 55\\*x\\*\\*9  + 1320\\*x\\*\\*8 - 18150\\*x\\*\\*7 + 157773\\*x\\*\\*6 - 902055\\*x\\*\\*5 + 3416930\\*x\\*\\*4 - 8409500\\*x\\*\\*3 + 12753576\\*x\\*\\*2 - 10628640\\*x + 3628800)", "done": true}︡
︠c21cd7de-d136-427b-87d6-99d8ff4f40c1i︠
%md
<img src="https://raw.githubusercontent.com//cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram3.jpg" />
︡4a0fae45-d873-4e5b-b62b-ef9c7d6268ae︡{"md": "<img src=\"https://raw.githubusercontent.com//cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram3.jpg\" />", "done": true}︡
︠69022ff5-1bfc-4176-a61a-33fd5d08aaeb︠
from sympy import *
x = Symbol('x')
factor(x**10 - 55*x**9  + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800)
︡5e8ca93a-4d99-46cc-8c12-557e6bb8f0d8︡{"stdout": "(x - 10)*(x - 9)*(x - 8)*(x - 7)*(x - 6)*(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)", "done": true}︡
︠73c0a147-c24e-4fdc-8a45-7fd755c20b3fi︠
%md
## SymPy’s solvers
- https://docs.sympy.org/latest/tutorial/solvers.html

- x\*\*10 - 55\*x\*\*9  + 1320\*x\*\*8 - 18150\*x\*\*7 + 157773\*x\*\*6 - 902055\*x\*\*5 + 3416930\*x\*\*4 - 8409500\*x\*\*3 + 12753576\*x\*\*2 - 10628640\*x + 3628800 = 0
︡c7f26f0a-af24-476c-a15a-a2663a96f1f9︡{"md": "## SymPy\u2019s solvers\n- https://docs.sympy.org/latest/tutorial/solvers.html\n\n- x\\*\\*10 - 55\\*x\\*\\*9  + 1320\\*x\\*\\*8 - 18150\\*x\\*\\*7 + 157773\\*x\\*\\*6 - 902055\\*x\\*\\*5 + 3416930\\*x\\*\\*4 - 8409500\\*x\\*\\*3 + 12753576\\*x\\*\\*2 - 10628640\\*x + 3628800 = 0", "done": true}︡
︠46a31411-040b-4549-a776-899cb8fd49a8i︠
%md
<img src="https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram4.jpg" />
︡b1861069-88d7-4556-bd34-22ebac2f7f7e︡{"md": "<img src=\"https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram4.jpg\" />", "done": true}︡
︠aab4bcaa-72dc-4447-aacf-a78ae2bac6d1︠
from sympy import *
x = Symbol('x')
solveset(Eq(x**10 - 55*x**9  + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800, 0), x)
︡683bee0b-d673-4d47-87c3-59bb03468214︡{"stdout": "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}", "done": true}︡
︠01ac8ae5-a5a3-496a-8343-f2075afac8b7i︠
%md
## SymPy’s Symbolic and Numercical Complex Evaluations
- https://docs.sympy.org/latest/modules/evalf.html](https://)
- x = x2 - I*x1,y = y2 - I*y1, z = z2 - I*z1, x*y*z
︡c184fe17-b405-48df-a200-bb1ae03f47e0︡{"md": "## SymPy\u2019s Symbolic and Numercical Complex Evaluations\n- https://docs.sympy.org/latest/modules/evalf.html](https://)\n- x = x2 - I*x1,y = y2 - I*y1, z = z2 - I*z1, x*y*z", "done": true}︡
︠74f99cd1-ede7-4c37-ac83-65ca49d7d688i︠
%md
<img src="https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram5.jpg" />
︡e0b09091-a24b-48b9-8683-23b9721f8ea3︡{"md": "<img src=\"https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram5.jpg\" />", "done": true}︡
︠25fff419-efff-43dc-b2bd-c16530671461︠
from sympy import *
x1, x2, y1, y2, z1, z2 = symbols("x1 x2 y1 y2 z1 z2", real=True)  
x = x2 - I*x1
y = y2 - I*y1
z = z2 - I*z1

print(x*y*z)
print(expand(x*y*z))
print(expand((x*y)*z))
print(expand(x*(y*z)))

w=N(1/(pi + I), 20)
print('w=',w)
︡d23e2656-7297-4b4a-a313-cf1bbd426838︡{"html": "<pre><span style=\"font-family:monospace;\">(-I*x1 + x2)*(-I*y1 + y2)*(-I*z1 + z2)\nI*x1*y1*z1 - x1*y1*z2 - x1*y2*z1 - I*x1*y2*z2 - x2*y1*z1 - I*x2*y1*z2 - I*x2*y2*z1 + x2*y2*z2\nI*x1*y1*z1 - x1*y1*z2 - x1*y2*z1 - I*x1*y2*z2 - x2*y1*z1 - I*x2*y1*z2 - I*x2*y2*z1 + x2*y2*z2\nI*x1*y1*z1 - x1*y1*z2 - x1*y2*z1 - I*x1*y2*z2 - x2*y1*z1 - I*x2*y1*z2 - I*x2*y2*z1 + x2*y2*z2\nw= 0.28902548222223624241 - 0.091999668350375232456*I\n</span></pre>", "done": true}︡
︠dadcac18-e155-4579-bbeb-6ffe871e0d0bi︠
%md
## SymPy’s integrals
- https://docs.sympy.org/latest/modules/integrals/integrals.html
- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)
︡ecee4794-04ce-4d42-af29-8a2d6dec2f80︡{"md": "## SymPy\u2019s integrals\n- https://docs.sympy.org/latest/modules/integrals/integrals.html\n- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)", "done": true}︡
︠9ed0de45-1910-41c9-b9e3-3fc0a0225420i︠
%md
Let’s start with a simple integration problem in 1D,

$$\int_1^5 x^2 dx$$
 
This is easy to solve analytically, and we can use the SymPy library in case you’ve forgotten how to resolve simple integrals.
︡8b8f46f8-5100-4ffa-9e32-6898feb94333︡{"md": "Let\u2019s start with a simple integration problem in 1D,\n\n$$\\int_1^5 x^2 dx$$\n \nThis is easy to solve analytically, and we can use the SymPy library in case you\u2019ve forgotten how to resolve simple integrals.", "done": true}︡
︠902103f1-38e6-4b06-af18-1d8ec66f052b︠
import sympy
# we’ll save results using different methods in this data structure, called a dictionary
result = {}  
x = sympy.Symbol("x")
i = sympy.integrate(x**2)
print(i)
result["analytical"] = float(i.subs(x, 5) - i.subs(x, 1))
print("Analytical result: {}".format(result["analytical"]))
︡c47fe9f3-671c-4ad0-baec-ceb2fa10e1a0︡{"html": "<pre><span style=\"font-family:monospace;\">x**3/3\nAnalytical result: 41.333333333333336\n</span></pre>", "done": true}︡
︠73991138-ba95-4f10-8f62-95d746e3aae1i︠
%md
**Integrating with Monte Carlo** 
[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) 

We can estimate this integral using a standard Monte Carlo method, where we use the fact that the expectation of a random variable is related to its integral

$$\mathbb{E}(f(x)) = \int_I f(x) dx $$

We will sample a large number N of points in I and calculate their average, and multiply by the range over which we are integrating.
︡176c9781-8e20-490a-b95b-b8de38cb3f9b︡{"md": "**Integrating with Monte Carlo** \n[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) \n\nWe can estimate this integral using a standard Monte Carlo method, where we use the fact that the expectation of a random variable is related to its integral\n\n$$\\mathbb{E}(f(x)) = \\int_I f(x) dx $$\n\nWe will sample a large number N of points in I and calculate their average, and multiply by the range over which we are integrating.", "done": true}︡
︠1acb9f64-d5fc-4af4-a3c9-d0eb45e97e6c︠
import numpy
N = 1_000_000
accum = 0
for i in range(N):
    x = numpy.random.uniform(1, 5)
    accum += x**2
volume = 5 - 1
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡57e03e7f-7ce0-40f7-855f-eda1267e093d︡{"html": "<pre><span style=\"font-family:monospace;\">Standard Monte Carlo result: 41.36348676222791\n</span></pre>", "done": true}︡
︠469c90ba-1dd8-4174-b23a-770be7e09635i︠
%md
- integrate(x\*\*2 * sin(x)\*\*3)

<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram8.jpg" />
︡35e0afbd-d6cd-4971-81af-327687a817d1︡{"md": "- integrate(x\\*\\*2 * sin(x)\\*\\*3)\n\n<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram8.jpg\" />", "done": true}︡
︠6cbd3b07-725b-4b3f-b467-3aca2a4b31da︠
import sympy
x = Symbol("x")
i = integrate(x**2 * sin(x)**3)
print(i)
print(float(i.subs(x, 5) - i.subs(x, 1)))
︡4e0138ed-0336-45dd-9f91-b6fe725bf032︡{"html": "<pre><span style=\"font-family:monospace;\">-x**2*sin(x)**2*cos(x) - 2*x**2*cos(x)**3/3 + 14*x*sin(x)**3/9 + 4*x*sin(x)*cos(x)**2/3 + 14*sin(x)**2*cos(x)/9 + 40*cos(x)**3/27\n-15.42978215330555\n</span></pre>", "done": true}︡
︠b411030a-75f6-4810-a23f-cd32bba3bd46︠
import numpy
N = 100_000
accum = 0
l =[]
for i in range(N):
    x = numpy.random.uniform(1, 5)
    accum += x**2 * sin(x)**3
volume = 5 - 1
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡c8ff7ac6-448f-4267-b710-388bb7e4858b︡{"html": "<pre><span style=\"font-family:monospace;\">Standard Monte Carlo result: -15.4984689174733\n</span></pre>", "done": true}︡
︠82ab7b9b-e03e-42ee-8743-1776198eda64i︠
%md
**A higher dimensional integral** [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) 


Let us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear.
︡59e8baf7-aee3-455a-8c67-3f46bf60a2f0︡{"md": "**A higher dimensional integral** [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) \n\n\nLet us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear.", "done": true}︡
︠3f26876f-cf23-4323-b140-0ff1eae2ad76︠
import sympy

x1 = sympy.Symbol("x1")
x2 = sympy.Symbol("x2")
x3 = sympy.Symbol("x3")
expr = sympy.sin(x1) + 7*sympy.sin(x2)**2 + 0.1 * x3**4 * sympy.sin(x1)
res = sympy.integrate(expr,
                      (x1, -sympy.pi, sympy.pi),
                      (x2, -sympy.pi, sympy.pi),
                      (x3, -sympy.pi, sympy.pi))
# Note: we use float(res) to convert res from symbolic form to floating point form
result = {} 
result["analytical"] = float(res)
print("Analytical result: {}".format(result["analytical"]))
︡c61f1fb9-e85a-4328-beaf-4befd02fd703︡{"html": "<pre><span style=\"font-family:monospace;\">Analytical result: 868.175747048395\n</span></pre>", "done": true}︡
︠a65e3314-d133-46e9-b3a5-eb7f58a5ab63︠
N = 10_000
accum = 0
for i in range(N):
    xx1 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx2 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx3 = numpy.random.uniform(-numpy.pi, numpy.pi)
    accum += numpy.sin(xx1) + 7*numpy.sin(xx2)**2 + 0.1 * xx3**4 * numpy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {} 
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡ebabffc9-9ed3-4192-b743-7ea002f21826︡{"html": "<pre><span style=\"font-family:monospace;\">Standard Monte Carlo result: 865.696142762518\n</span></pre>", "done": true}︡
︠12b6aef0-756c-4dab-9e35-963c3845a49c︠
import math
import numpy
# adapted from https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html
def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)
︡6226a5e2-f847-45f3-8363-36c25b6fb45b︡{"stdout": "", "done": true}︡
︠563d7956-6b32-4a2b-8bb0-c408475c6551︠
import matplotlib.pyplot as plt
N = 1000
seq = halton(2, N)
plt.title("2D Halton sequence")
# Note: we use "alpha=0.5" in the scatterplot so that the plotted points are semi-transparent
# (alpha-transparency of 0.5 out of 1), so that we can see when any points are superimposed.
plt.axes().set_aspect('equal')
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
︡dc70c18d-2497-422d-85aa-e3ab833a9a0f︡{"html": "<pre><span style=\"font-family:monospace;\">/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:7: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.\n  import sys\n</span></pre>", "done": true}︡
︠ad77ecd0-364d-4f7e-8e3f-394aad368029︠
N = 10_000

seq = halton(3, N)
accum = 0
for i in range(N):
    xx1 = -numpy.pi + seq[i][0] * numpy.pi * 2
    xx2 = -numpy.pi + seq[i][1] * numpy.pi * 2
    xx3 = -numpy.pi + seq[i][2] * numpy.pi * 2
    accum += numpy.sin(xx1) + 7*numpy.sin(xx2)**2 + 0.1 * xx3**4 * numpy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {} 
result["MC"] = volume * accum / float(N)
print("Qausi Monte Carlo Halton Sequence result: {}".format(result["MC"]))
︡1ee0e36e-effb-4c10-86ec-4a00b6dd28f8︡{"html": "<pre><span style=\"font-family:monospace;\">Qausi Monte Carlo Halton Sequence result: 868.238928030592\n</span></pre>", "done": true}︡
︠df5d28d2-c0b1-4ad6-b755-3ac7b4954d1di︠
%md
## Wolfram alpha answers quastion in natural languaje
- What frequencies can a guitar sound?

<img src="https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram6.jpg" />
︡f30ef238-d416-4383-8c47-a003a485e7b9︡{"md": "## Wolfram alpha answers quastion in natural languaje\n- What frequencies can a guitar sound?\n\n<img src=\"https://raw.githubusercontent.com/cdsaineac/AlgorithmsUN2020II/master/SympyLab/sympylabwolfram6.jpg\" />", "done": true}︡









