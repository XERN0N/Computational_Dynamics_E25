//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)


= Assignment 1 - Computational Dynamics


#figure(
  image("Intro_pic.png")
)


== Problem:


=== 1. Identify the number of bodies, joints and degrees of freedom for the mechanism.

The number of bodies (nb) is 3 and there are 4 joints with 1 degree of freedom each (nh = 8).

Therefore there degrees of freedom for the model is $3 dot "nb" - "nh" = 1$

The remaining DoF will be governed by a driving constraint $Phi^D (q,t) = phi.alt_1 - omega t, #h(0.5cm) omega = 1.5 frac("rad", "s")$\

=== 2. Setup the kinematic constraints $Phi^K$ for all the joints

First A and r are defined:

$A(phi.alt) = mat(delim: "[", cos(phi.alt), -sin(phi.alt); sin(phi.alt), cos(phi.alt)), #h(1em) r^p = r + A(phi.alt)s'^P, #h(1em) q = mat(delim: "[", x_1, y_1, phi.alt_1, x_2, y_2, phi.alt_2, x_3, y_3, phi.alt_3)^T$


The points for the joints are described by:

$
s_1^'^"p1" = mat(delim: "[", -frac(L_1, 2); 0),
#h(1em) s_1^'^"p2" = mat(delim: "[", frac(L_1, 2); 0),
#h(1em) s_2^'^"p2" = mat(delim: "[", -frac(L_2, 2); 0),
#h(1em) s_2^'^"p3" = mat(delim: "[", frac(L_2, 2); 0),
#h(1em) s_3^'^"p3" = mat(delim: "[", -frac(L_3, 2); 0),
#h(1em) s_3^'^"p4" = mat(delim: "[", frac(L_3, 2); 0),
 $

so that e.g. $s_1^'^"p2"$ describes the local point p2 in body 1.

The points 

The Kinematic constraint equations are as follows:

- $Phi^"abs1" (q) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^1))) = mat(delim: "[",
 x_1 - frac(L_1, 2)cos(phi.alt_1); 
 y_1 - frac(L_1, 2)sin(phi.alt_1)) 
 = mat(delim: "[", 0;0)$

- $Phi^"rel1" (q) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^2)) - bold(r_2 + bold(A"s"_2^'^P^2)))
 = mat(delim: "[",
 x_1 + frac(L_1, 2)cos(phi.alt_1) - x_2 -frac(L_2, 2)cos(phi.alt_2);
 y_1 + frac(L_1, 2)sin(phi.alt_1) - y_2 -frac(L_2, 2)sin(phi.alt_2))
 = mat(delim: "[", 0;0)$

- $Phi^"rel2" (q) = mat(delim: "[", bold(r_2 + bold(A"s"_2^'^P^3)) - bold(r_3 + bold(A"s"_3^'^P^3))) 
 = mat(delim: "[", 
 x_2 + frac(L_2, 2)cos(phi.alt_2) - x_3 -frac(L_3, 2)cos(phi.alt_3);
 y_2 + frac(L_2, 2)sin(phi.alt_2) - y_3 -frac(L_3, 2)sin(phi.alt_3))
 = mat(delim: "[", 0;0)$

- $Phi^"abs2" (q) = mat(delim: "[", bold(r_3 + bold(A"s"_3^'^P^4))) = mat(delim: "[",
 x_3 + frac(L_3, 2)cos(phi.alt_3);
 y_3 + frac(L_3, 2)sin(phi.alt_3))
 = mat(delim: "[", 0;0)$

As a system of constraints the vectorfunction the above equations become:

$Phi^K (q) = mat(delim: "[", Phi^"abs1" (q); Phi^"rev1" (q); Phi^"rev2" (q); Phi^"abs2" (q)) = mat(delim: "[",
 x_1 - frac(L_1, 2)cos(phi.alt_1); 
 y_1 - frac(L_1, 2)sin(phi.alt_1);
 x_1 + frac(L_1, 2)cos(phi.alt_1) - x_2 -frac(L_2, 2)cos(phi.alt_2);
 y_1 + frac(L_1, 2)sin(phi.alt_1) - y_2 -frac(L_2, 2)sin(phi.alt_2);
 x_2 + frac(L_2, 2)cos(phi.alt_2) - x_3 -frac(L_3, 2)cos(phi.alt_3);
 y_2 + frac(L_2, 2)sin(phi.alt_2) - y_3 -frac(L_3, 2)sin(phi.alt_3);
 x_3 + frac(L_3, 2)cos(phi.alt_3);
 y_3 + frac(L_3, 2)sin(phi.alt_3);)
 = mat(delim: "[", 0;0;0;0;0;0;0;0;0) = mat(delim: "[", arrow.r(o))$


=== 3. Setup of the driving constraint $Phi^D$ that imposes $phi.alt_1 = omega t, #h(1em) omega = 1.5 frac("rad", "s")$
#v(0.5em) As there is 1 DoF for the system, an absolute driving constraint is added to the system:

$Phi^D (q,t) = mat(delim: "[", phi.alt_1 - omega t) = 0$

=== 4. Calculate the constraint jacobian $Phi_q$
by combining $Phi^K$ and $Phi^D$ into $Phi$ and then taking the partial derivative with respect to $q$ the constraint jacobian $Phi_q$ can be obtained:

$bold(Phi_q) = ded(Phi,q) = mat(delim: "[", ded(Phi^K, q); ded(Phi^D, q))
=
mat(delim: "[", 
I_"2x2",  B_1s'^"p1", 0, 0, 0, 0,;
I_"2x2", -B_1s'^"p1", -I_"2x2", B_2s'^"p2", 0, 0;
0, 0, I_"2x2", - B_2s'^"p2", -I_"2x2", B_3s'^"p3";
0, 0, 0, 0, I_"2x2", - B_3s'^"p3";
0, 1, 0, 0, 0, 0;
)\
#h(8.4em) =
mat(delim: "[",
1, 0, frac(L_1, 2)sin(phi.alt_1), 0, 0, 0, 0, 0, 0;
0, 1, -frac(L_1, 2)cos(phi.alt_1), 0, 0, 0, 0, 0, 0;
1, 0, -frac(L_1, 2)sin(phi.alt_1), -1, 0, frac(L_2, 2)sin(phi.alt_2), 0, 0, 0;
0, 1, frac(L_1, 2)cos(phi.alt_1), 0, -1, -frac(L_2, 2)cos(phi.alt_2), 0, 0, 0;
0, 0, 0, 1, 0, -frac(L_2, 2)sin(phi.alt_2), -1, 0, frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 1, frac(L_2, 2)cos(phi.alt_2), 0, -1, -frac(L_3, 2)cos(phi.alt_3);
0, 0, 0, 0, 0, 0, 1, 0, -frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 0, 0, 0, 1, frac(L_3, 2)cos(phi.alt_3);
0, 0, 1, 0, 0, 0, 0, 0, 0;
)
$

=== 5. Setup of the velocity and acceleration equations $nu$ and $gamma$
#v(0.5em) The velocity equations are defined as:

$bold(nu) = bold(-Phi_t) = bold(Phi_q dot(q))$
#h(1em) where $bold(dot(q)) = mat(delim: "[", dot(x_1), dot(y_1), dot(phi.alt_1), dot(x_2), dot(y_2), dot(phi.alt_2), dot(x_3), dot(y_3), dot(phi.alt_3))^T$

$bold(nu) = mat(delim: "[", 0, 0, 0, 0, 0, 0, 0, 0, w)^T = bold(Phi_q dot(q))$

By left multiplying with $bold(Phi_q)^(-1)$ the equation can be rewritten as to obtain $bold(dot(q))$:

$bold(dot(q) = bold(Phi_q)^(-1)bold(nu))$

The acceleration equations are defined as:

$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)) - 2 bold(Phi)_(bold(q)t)bold(dot(q)) - bold(Phi)_"tt"$
#h(1em) where $bold(dot.double(q)) = mat(delim: "[", dot.double(x_1), dot.double(y_1), dot.double(phi.alt_1), dot.double(x_2), dot.double(y_2), dot.double(phi.alt_2), dot.double(x_3), dot.double(y_3), dot.double(phi.alt_3))^T$

As $bold(Phi_q)$ does not contain $t$, $bold(Phi)_(bold(q)"t") = [arrow.r(0)]$ and as only $bold(Phi)^D$ is dependant on time:

$bold(Phi)_"tt" = frac(diff^2bold(Phi)^D, diff^2t) = ded(bold(Phi)_t^D, t) = [arrow.r(0)]$

The expression for $bold(gamma)$ then becomes:
$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)),$ where when solving for $bold(dot.double(q))$:



$bold(Phi_q dot(q)) = mat(delim: "[", 
I_"2x2", dot(phi.alt_1) B_1s'^"p1", 0, 0, 0, 0,;
I_"2x2", -dot(phi.alt_1) B_1s'^"p1", -I_"2x2", dot(phi.alt_2) B_2s'^"p2", 0, 0;
0, 0, I_"2x2", -dot(phi.alt_2) B_2s'^"p2", -I_"2x2", dot(phi.alt_3) B_3s'^"p3";
0, 0, 0, 0, I_"2x2", -dot(phi.alt_3) B_3s'^"p3";
0, 1, 0, 0, 0, 0;
)
mat(delim: "[",
 bold(dot(r_1)); dot(phi.alt_1); bold(dot(r_2)); dot(phi.alt_2); bold(dot(r_3)); dot(phi.alt_3))^T =
$



$mat(delim: "[", 
I_"2x2", dot.double(phi.alt_1) B_1s'^"p1"-dot(phi.alt_1)^2A_1s'^"p1", 0, 0, 0, 0,;
I_"2x2", -dot.double(phi.alt_1) B_1s'^"p1"+dot(phi.alt_1)^2A_1s'^"p1", -I_"2x2", dot.double(phi.alt_2) B_2s'^"p2"-dot(phi.alt_2)^2A_2s'^"p2", 0, 0;
0, 0, I_"2x2", -dot.double(phi.alt_2) B_2s'^"p2"+dot(phi.alt_2)^2A_2s'^"p2", -I_"2x2", dot.double(phi.alt_3) B_3s'^"p3"-dot(phi.alt_2)^2A_3s'^"p3";
0, 0, 0, 0, I_"2x2", -dot.double(phi.alt_3) B_3s'^"p3"+dot(phi.alt_2)^2A_3s'^"p3";
0, 1, 0, 0, 0, 0;
)
mat(delim: "[",
 bold(dot.double(r_1)); dot.double(phi.alt_1); bold(dot.double(r_2)); dot.double(phi.alt_2); bold(dot.double(r_3)); dot.double(phi.alt_3))^T = $

$- mat(delim: "[",
dot(x_1), 0, dot(phi.alt_1)^2frac(L_1, 2)sin(phi.alt_1), 0, 0, 0, 0, 0, 0;
0, dot(y_1), -dot(phi.alt_1)^2frac(L_1, 2)cos(phi.alt_1), 0, 0, 0, 0, 0, 0;
dot(x_1), 0, -dot(phi.alt_1)^2frac(L_1, 2)sin(phi.alt_1), -dot(x_2), 0, dot(phi.alt_2)^2frac(L_2, 2)sin(phi.alt_2), 0, 0, 0;
0, dot(y_1), dot(phi.alt_1)^2frac(L_1, 2)cos(phi.alt_1), 0, -dot(y_2), -dot(phi.alt_2)^2frac(L_2, 2)cos(phi.alt_2), 0, 0, 0;
0, 0, 0, dot(x_2), 0, -dot(phi.alt_2)^2frac(L_2, 2)sin(phi.alt_2), -dot(x_3), 0, dot(phi.alt_3)^2frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, dot(y_2), dot(phi.alt_2)^2frac(L_2, 2)cos(phi.alt_2), 0, -dot(y_3), -dot(phi.alt_3)^2frac(L_3, 2)cos(phi.alt_3);
0, 0, 0, 0, 0, 0, dot(x_3), 0, -dot(phi.alt_3)^2frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 0, 0, 0, dot(y_3), dot(phi.alt_3)^2frac(L_3, 2)cos(phi.alt_3);
0, 0, dot(phi.alt_1), 0, 0, 0, 0, 0, 0;
)\ -mat(delim: "[",
0, 0, a, 0, 0, 0 ,0 ,0 ,0;
0, 0, a, 0, 0, 0 ,0 ,0 ,0;
0, 0, a, 0, 0, b ,0 ,0 ,0;
0, 0, a, 0, 0, b ,0 ,0 ,0;
0, 0, 0, 0, 0, b ,0 ,0 ,c;
0, 0, 0, 0, 0, b ,0 ,0 ,c;
)$

 


$bold(dot.double(q)) = bold(Phi_q)^(-1)(bold(-(Phi_q dot(q))_q dot(q)))$





Trash: \*

$bold(Phi_q) = ded(Phi,q) = mat(delim: "[", ded(Phi^K, q); ded(Phi^D, q))
=
mat(delim: "[", 
I_"2x2", dot(phi.alt_1) B_1s'^"p1", 0, 0, 0, 0,;
I_"2x2", -dot(phi.alt_1) B_1s'^"p1", -I_"2x2", dot(phi.alt_2) B_2s'^"p2", 0, 0;
0, 0, I_"2x2", -dot(phi.alt_2) B_2s'^"p2", -I_"2x2", dot(phi.alt_3) B_3s'^"p3";
0, 0, 0, 0, I_"2x2", -dot(phi.alt_3) B_3s'^"p3";
0, 1, 0, 0, 0, 0;
)\
#h(8.4em) =
mat(delim: "[",
1, 0, dot(phi.alt_1)frac(L_1, 2)sin(phi.alt_1), 0, 0, 0, 0, 0, 0;
0, 1, -dot(phi.alt_1)frac(L_1, 2)cos(phi.alt_1), 0, 0, 0, 0, 0, 0;
1, 0, -dot(phi.alt_1)frac(L_1, 2)sin(phi.alt_1), -1, 0, dot(phi.alt_2)frac(L_2, 2)sin(phi.alt_2), 0, 0, 0;
0, 1, dot(phi.alt_1)frac(L_1, 2)cos(phi.alt_1), 0, -1, -dot(phi.alt_2)frac(L_2, 2)cos(phi.alt_2), 0, 0, 0;
0, 0, 0, 1, 0, -dot(phi.alt_2)frac(L_2, 2)sin(phi.alt_2), -1, 0, dot(phi.alt_3)frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 1, dot(phi.alt_2)frac(L_2, 2)cos(phi.alt_2), 0, -1, -dot(phi.alt_3)frac(L_3, 2)cos(phi.alt_3);
0, 0, 0, 0, 0, 0, 1, 0, -dot(phi.alt_3)frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 0, 0, 0, 1, dot(phi.alt_3)frac(L_3, 2)cos(phi.alt_3);
0, 0, 1, 0, 0, 0, 0, 0, 0;
)
$



$mat(delim: "[",
dot(x_1), 0, dot(phi.alt_1)frac(L_1, 2)sin(phi.alt_1), 0, 0, 0, 0, 0, 0;
0, dot(y_1), -dot(phi.alt_1)frac(L_1, 2)cos(phi.alt_1), 0, 0, 0, 0, 0, 0;
dot(x_1), 0, -dot(phi.alt_1)frac(L_1, 2)sin(phi.alt_1), -dot(x_2), 0, dot(phi.alt_2)frac(L_2, 2)sin(phi.alt_2), 0, 0, 0;
0, dot(y_1), dot(phi.alt_1)frac(L_1, 2)cos(phi.alt_1), 0, -dot(y_2), -dot(phi.alt_2)frac(L_2, 2)cos(phi.alt_2), 0, 0, 0;
0, 0, 0, dot(x_2), 0, -dot(phi.alt_2)frac(L_2, 2)sin(phi.alt_2), -dot(x_3), 0, dot(phi.alt_3)frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, dot(y_2), dot(phi.alt_2)frac(L_2, 2)cos(phi.alt_2), 0, -dot(y_3), -dot(phi.alt_3)frac(L_3, 2)cos(phi.alt_3);
0, 0, 0, 0, 0, 0, dot(x_3), 0, -dot(phi.alt_3)frac(L_3, 2)sin(phi.alt_3);
0, 0, 0, 0, 0, 0, 0, dot(y_3), dot(phi.alt_3)frac(L_3, 2)cos(phi.alt_3);
0, 0, dot(phi.alt_1), 0, 0, 0, 0, 0, 0;
)$