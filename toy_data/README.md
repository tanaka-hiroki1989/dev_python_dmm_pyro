$\{x_t\}_{1,2,\dots,T}$ : 観測変数, 3次元

$\{z_t\}_{1,2,\dots,T}$ : 状態変数, 2次元
$$
\begin{align*}
z_{t+1}&=Az_t+\epsilon\\
x_{t}&=Bz_t+\epsilon'
\end{align*}
$$
初期状態
$$
\begin{align*}
z_1&=\begin{bmatrix}
1\\0
\end{bmatrix}
\end{align*}
$$
$$
\begin{align*}
A&=\begin{bmatrix}
0.9&0.1\\
0.1&0.9
\end{bmatrix},
B&=\begin{bmatrix}
0.7&0.3\\
0.8&0.2\\
0.2&0.8
\end{bmatrix}

\end{align*}
$$