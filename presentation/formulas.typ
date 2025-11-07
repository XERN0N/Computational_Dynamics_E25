#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

//CONSTRAINT EQUATIONS
$bold(Phi)^t(i,j)=
mat(delim: "[",
(bold(v)_i^perp)bold(d)_"ij";
(bold(v)_i^perp)^T bold(v)_j) = mat(delim: "[",
bold(v)_i^'^T bold(B)_i^T (bold(r)_j-bold(r)_i) - bold(v)_i^'^T bold(B)_"ij"bold(s)_j^'^p- bold(v)_i^'^T bold(R)^T bold(s)_i^'^P;
- bold(v)_i^'^T bold(B)_"ij" bold(v)_j^') = 0$

//CONSTRAINT JACOBIAN QI
$bold(Phi_q)_i^t(i,j) = mat(delim: "[",
-bold(v)_i^'^T bold(B)_i^T, - bold(v)_i^'^T bold(A)_i^T (bold(r)_j-bold(r)_i)  - bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, - bold(v)_i^'^T bold(A)_"ij" bold(v)_j^')$

//CONSTRAINT JACOBIAN QJ
$bold(Phi_q)_j^t(i,j) = mat(delim: "[",
bold(v)_i^'^T bold(B)_i^T, bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, bold(v)_i^'^T bold(A)_"ij" bold(v)_j^')$

//CONSTRAINT JACOBIAN CONCATENATED
$bold(Phi_q)^t(i,j) = mat(delim: "[", bold(v)_i^'^T bold(B)_i^T, bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p,
-bold(v)_i^'^T bold(B)_i^T, - bold(v)_i^'^T bold(A)_i^T (bold(r)_j-bold(r)_i)  - bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, bold(v)_i^'^T bold(A)_"ij" bold(v)_j^', 0, -bold(v)_i^'^T bold(A)_"ij" bold(v)_j^')$

//CONSTRAINT Q X Q_DOT

$bold(Phi_q)^t(i,j) bold(dot(q))= mat(delim: "[", bold(v)_i^'^T bold(B)_i^T, bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p,
-bold(v)_i^'^T bold(B)_i^T, - bold(v)_i^'^T bold(A)_i^T (bold(r)_j-bold(r)_i)  - bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, bold(v)_i^'^T bold(A)_"ij" bold(v)_j^', 0, -bold(v)_i^'^T bold(A)_"ij" bold(v)_j^') mat(delim: "[",
bold(dot(r))_i;
bold(dot(phi.alt))_i;
bold(dot(r))_j;
bold(dot(phi.alt))_j;
)$

#v(1em)$= mat(delim: "[",
0;
0;
0;
0;
)$


//GAMMA

$bold(gamma)^t(i,j) = - mat(delim: "[",
bold(v)_i^'^T [bold(B)_"ij"bold(s)_j^'^p (bold(dot(phi.alt_j)-dot(phi.alt_i)))^2 - bold(B)_i^T (bold(dot(r)_j) - bold(dot(r)_i))bold(dot(phi.alt))_i^2 - 2 bold(A)_i^T (bold(dot(r)_j - bold(dot(r)_i)))bold(dot(phi.alt))_i];
0
)$