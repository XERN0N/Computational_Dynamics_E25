#set page(header: [
  _Sigurd Mousten Jager Nielsen_ #h(1fr) Aarhus Universitet - Computational Dynamics E25
])

#set page(numbering: "1 of 1")

//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)


#figure(
  image("image.png"))

= 1.
In this section the kinematic analysis will be performed. This section has some overlap with assignment 1, which has been repeated here.

== 1.1 DoF for the mechanism

There are 3 bodies in the mechanism with each 3 DoF. There are also 4 joints: 3 revolute joints and 1 translational joint. There is also 1 driver with constant rotational speed.

$"DoF" = "nb" dot 3 - "nh" = 3 dot 3 - (4 dot 2 + 1 dot 1) = 0$

== 1.2 The Joints:
To describe the joints A and r are defined:

$ bold(A)(phi.alt) = mat(delim: "[", cos(phi.alt), -sin(phi.alt); sin(phi.alt), cos(phi.alt)) in RR^("2x2"), #h(0.5em) bold(r^p = r + A)(phi.alt)bold(s'^P) in RR^2 $

$ bold(q) = mat(delim: "[", x_1, y_1, phi.alt_1, x_2, y_2, phi.alt_2, x_3, y_3, phi.alt_3)^T in RR^9 $


*Joints A and D*

The joints between the bodies 1, 3 and ground at point A and D are both absolute position joints and are described by the following equations:

$ bold(Phi^"absA" (q)) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^A - C_1))) = mat(delim: "[", 0;0) $

$ bold(Phi^"absD" (q)) = mat(delim: "[", bold(r_3 + bold(A"s"_3^'^P^D - C_3))) = mat(delim: "[", 0;0) $

The jacobian entries of the absolute position joint are in general:
//CONSTRAINT EQUATIONS IJ
$ bold(Phi)^"abs(i)"=
mat(delim: "[", I_"2x2", bold(B)_i bold("s")_i^'^P) $

And for the joints A and D:

//CONSTRAINT EQUATIONS A
$ bold(Phi)^"absA"=
mat(delim: "[", I_"2x2", bold(B)_1 bold("s")_1^'^P^A) $

//CONSTRAINT EQUATIONS D
$ bold(Phi)^"absD"=
mat(delim: "[", I_"2x2", bold(B)_3 bold("s")_3^'^P^D) $

*Joint B*

The joint B between body 1 and 2 at point B is a revolute joint is described by the following equations:

$ bold(Phi^"revB" (q)) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^B)) - (bold(r_2 + bold(A"s"_2^'^P^B))))
 = mat(delim: "[", 0;0) $

The jacobian entries for this joint is 

And for the joints A and D:

//CONSTRAINT EQUATIONS B
$ bold(Phi_q)^"revB"=
mat(delim: "[", I_"2x2", bold(B)_1 bold("s")_1^'^P^B, -I_"2x2", -bold(B)_2 bold("s")_2^'^P^B) $

*Joint C*

The joint C between at point B is a revolute joint  are both absolute position joints and are in general described by the following equations:

//CONSTRAINT EQUATIONS
$ bold(Phi)^t(i,j)bold((q))=
mat(delim: "[",
(bold(v)_i^perp)bold(d)_"ij";
(bold(v)_i^perp)^T bold(v)_j) = mat(delim: "[",
bold(v)_i^'^T bold(B)_i^T (bold(r)_j-bold(r)_i) - bold(v)_i^'^T bold(B)_"ij"bold(s)_j^'^p- bold(v)_i^'^T bold(R)^T bold(s)_i^'^P;
- bold(v)_i^'^T bold(B)_"ij" bold(v)_j^') = mat(delim: "[", 0;0) $

And for this joint C:
//CONSTRAINT EQUATIONS inserted points
$ bold(Phi)^"C,t"(2,3)bold((q))=
mat(delim: "[",
(bold(v)_2^perp)bold(d)_"23";
(bold(v)_2^perp)^T bold(v)_3) = mat(delim: "[",
bold(v)_2^'^T bold(B)_2^T (bold(r)_3-bold(r)_2) - bold(v)_2^'^T bold(B)_"23"bold(s)_3^'^"pC"- bold(v)_2^'^T bold(R)^T bold(s)_2^'^"pC";
- bold(v)_2^'^T bold(B)_"23" bold(v)_3^') = mat(delim: "[", 0;0) $

The Constraint jacobian entry for body i and j respectively then becomes:
//CONSTRAINT JACOBIAN QI
$ bold(Phi_q)_i^t(i,j) = mat(delim: "[",
-bold(v)_i^'^T bold(B)_i^T, - bold(v)_i^'^T bold(A)_i^T (bold(r)_j-bold(r)_i)  - bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, - bold(v)_i^'^T bold(A)_"ij" bold(v)_j^') $

//CONSTRAINT JACOBIAN QJ
$ bold(Phi_q)_j^t(i,j) = mat(delim: "[",
bold(v)_i^'^T bold(B)_i^T, bold(v)_i^'^T bold(A)_"ij"bold(s)_j^'^p;
0, bold(v)_i^'^T bold(A)_"ij" bold(v)_j^') $

The constraint jacobian for the translational joint C then becomes:
//CONSTRAINT JACOBIAN CONCATENATED
$ bold(Phi_q)^"C,t"(2,3) = mat(delim: "[", bold(v)_2^'^T bold(B)_2^T, bold(v)_2^'^T bold(A)_"23"bold(s)_3^'^"pC",
-bold(v)_2^'^T bold(B)_2^T, - bold(v)_2^'^T bold(A)_2^T (bold(r)_3-bold(r)_2)  - bold(v)_2^'^T bold(A)_"23"bold(s)_3^'^"pC";
0, bold(v)_2^'^T bold(A)_"23" bold(v)_3^', 0, -bold(v)_2^'^T bold(A)_"23" bold(v)_3^') $

*The driving constraint*

The driving constraint is defined as:
$ Phi^D (q,t) = phi.alt_1 - omega t, #h(0.5cm) omega = 1.5 frac("rad", "s") $
The jacobian entry is

$ Phi^D (q,t) = I_"1x1" $

== 1.2 The constraint equations $bold(Phi)$
The constraint equations are all the joints and the driver:

$ bold(Phi) (bold(q),t) = mat(delim: "[", 
bold(r_1 + bold(A"s"_1^'^P^A - C_1));
bold(r_3 + bold(A"s"_3^'^P^D - C_3));
bold(r_1 + bold(A"s"_1^'^P^B)) - (bold(r_2 + bold(A"s"_2^'^P^B)))- bold(C_2);
bold(v)_2^'^T bold(B)_2^T (bold(r)_3-bold(r)_2) - bold(v)_2^'^T bold(B)_"23"bold(s)_3^'^"pC"- bold(v)_2^'^T bold(R)^T bold(s)_2^'^"pC";
phi.alt_1 - omega t
) $


== 1.3 The constraint jacobian $bold(Phi_q)$
The constraint jacobian can now be assembled using the jacobian entries from each joint:

$ bold(Phi_q) =  mat(delim: "[",
 bold(Phi)^"absA",0_"2x3",0_"2x3";
 bold(Phi)^"revB", -bold(Phi)^"revB", 0_"2x3";  
 0_"2x3", -bold(Phi_q)^"C,t"(2,3), bold(Phi_q)^"C,t"(2,3);  
 0_"2x3",0_"2x3", bold(Phi)^"absD";
 [0,0,1], 0_"1x3", 0_"1x3";
 ) $

== 1.4 Position analysis
To get the positions for the mechanism at each time instance the constraint jacobian $bold(Phi_q)$ is used in conjunction with the constraint equations $bold(Phi)$ and then finding the roots using the newton-rhapson algorithm.

The positions of each body is plotted in the figure below.

#figure(
  image("states_positions.png")
)


== 1.5 Velocity analysis
The velocity equation is defined as:

$bold(nu) = bold(-Phi)_t = bold(Phi_q dot(q)),$
#h(1em) where $bold(dot(q)) = mat(delim: "[", dot(r); dot(phi.alt))$

as only $bold(Phi)^D$ is a function of time $(phi.alt_1-omega t)$ the vector $bold(-Phi)_t$ becomes:

$bold(nu) = mat(delim: "[", 0, 0, 0, 0, 0, 0, 0, 0, w)^T = bold(Phi_q dot(q))$

By left multiplying with $bold(Phi_q)^(-1)$ the equation can be rewritten as to obtain $bold(dot(q))$:

$bold(dot(q) = bold(Phi_q)^(-1)bold(nu))$

for which the velocities can be calculated for each time instance.

The velocities of each body is plotted in the figure below.

#figure(
  image("states_velocities.png")
)


== 1.6 Acceleration analysis
The acceleration equations are defined as:

$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)) - 2 bold(Phi)_(bold(q)t)bold(dot(q)) - bold(Phi)_"tt",$
#h(1em) where $bold(dot.double(q)) = mat(delim: "[", dot.double(r); dot.double(phi.alt))$

As $bold(Phi_q)$ does not contain $t$, $bold(Phi)_(bold(q)"t") = [bold(arrow.r(0))]$ and as only $bold(Phi)^D$ is dependant on time:

$bold(Phi)_"tt" = frac(diff^2bold(Phi)^D, diff^2t) = ded(bold(Phi)_t^D, t) = [arrow.r(0)]$

The absolute position joints and relative rotation joint has been described in assignment 1 so the only new joint type is the translational joint for which $gamma$ is:

//GAMMA

$ bold(gamma)^t(i,j) = - mat(delim: "[",
bold(v)_i^'^T [bold(B)_"ij"bold(s)_j^'^p (bold(dot(phi.alt_j)-dot(phi.alt_i)))^2 - bold(B)_i^T (bold(dot(r)_j) - bold(dot(r)_i))bold(dot(phi.alt))_i^2 - 2 bold(A)_i^T (bold(dot(r)_j - bold(dot(r)_i)))bold(dot(phi.alt))_i];
0
) $
Please note that this is from Haug's book 2nd edition below equation 3.3.14 where the parenthesis error is fixed.

The expression for $bold(gamma)$ then becomes:
$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)),$ where when solving for $bold(dot.double(q))$:

$bold(dot.double(q)) = bold(Phi_q)^(-1)(bold(-(Phi_q dot(q))_q dot(q)))$

$bold(gamma)$ for this system then becomes:

$ bold(gamma) = mat(delim: "[",
dot(phi.alt)_1^2bold(A)_1 bold(s)^('A);
dot(phi.alt)_1^2bold(A)_1 bold(s)^('B)-dot(phi.alt)_2^2bold(A)_2 bold(s)^('B);
-bold(v)_2^'^T [bold(B)_"23"bold(s)_3^'^p (bold(dot(phi.alt_3)-dot(phi.alt_2)))^2 - bold(B)_2^T (bold(dot(r)_3) - bold(dot(r)_2))bold(dot(phi.alt))_2^2 - 2 bold(A)_2^T (bold(dot(r)_3 - bold(dot(r)_2)))bold(dot(phi.alt))_2];
0;
dot(phi.alt)_3^2bold(A)_3 bold(s)^('D);
0;
) $

The accelerations of each body is plotted in the figure below.

#figure(
  image("states_accelerations.png")
)

== 1.7 The mechanism and trajectory plotted
The trajectories of the mechanism can be seen in the figure.

#figure(
  image("mechanism_first_frame.png")
)

#figure(
  image("mechanism_last_frame.png")
)

= 2. Accelerations at point E
In this section the accelerations at point E will be calculated.

== 2.1 Describing the accelerations at point E
The accelerations at point E can be calculated by using the following equation:
$ bold(dot.double(r))^E = bold(dot.double(r))_3 + bold(dot.double(phi.alt))_3bold(B)_3bold(s)'^E - bold(dot(phi.alt))_3^2bold(A)_3bold(s)'^E $
And as $bold(dot.double(r))_3, bold(dot.double(phi.alt))_3, bold(dot(phi.alt))_3, bold(phi.alt)_3, bold(s)'^E$ are calculated prior in section 1 then all the values of the variables to each timestep are known.

== 2.2 Plots of the accelerations at point E
The figure below shows each component of accelerations at point E.

#figure(
  image("point_E_accel_components.png")
)

The Figure below shows the magnitude of the accelerations at point E, where it is notable that the accelerations of about $66 m/s^2$ is approximately 6.7 times larger than the acceleration due to gravitational forces. It happens approximately when the body 2 is closer to point D which makes sense as the prescribed movement of the driver will cause a larger acceleration at E when the mechanical advantage is higher.
#figure(
  image("point_E_accel_magnitude.png")
)

= 3. Driver torque using inverse Dynamics
In this section the driver torque will be solved for using inverse dynamics and then plotted.

== 3.1 Equations of motion using DAE's
By using equation 6.3.18 in Haug's book (2nd ed.) the matrix form of the equation is:


$ mat(delim: "[", 
bold(M), bold(Phi_q)^T;
bold(Phi_q), bold(0)
) mat(delim: "[",
bold(dot.double(q));
bold(lambda)) = mat(delim: "[",
bold(Q)^A;
bold(gamma)
) $

where $bold(M)$ is the positive definite mass matrix of the system, $bold(Phi_q)$ is the constraint jacobian, $bold(lambda)$ is the Lagrange multiplier (vector) for the system, $bold(Q)^A$ is the generalized applied forces and $bold(gamma)$ used in the acceleration equation.

for the system the mass matrix will be a diagonal matrix of sizes $3 dot "nb" times 3 dot "nb" = 9 times 9$. For a single body it would be $ mat(delim: "[", m_1, 0, 0;
0, m_1, 0;
0, 0, J^') $
Where $m_1$ and $ J^'$ describe the mass of the element and the local inertia of the body.

For the generalized applied forces: $bold(Q)^A$ the only forces acting on the bodies will be gravity:

$ bold(Q)^A = mat(delim: "[", 0, -m_1 dot g, 0, 0, -m_2 dot g, 0, ... , 0, -m_"nb" dot g, 0)^T $


== 3.2 Solving the equations of motion
To be able to solve the equations of motion the masses and moments of the bodies must be calculated.

In the description the masses are:

$ m_1 = 40 "kg", #v(0.5em) m_2 = 20 "kg", #v(0.5em) m_3 = 160 "kg" $

The bodies 1 and 3 are assumed to be approximately thin rods and body 2 a rectangle using the formulas in table 6.11 in Haug's book (2nd edition).

$ J'_"thin rod" = m/12 l^2, #v(0.5em) J'_"rectangle" = 1/12 m(a^2 + b^2) $
Body 2 is assumed per the hint in the description to have the width and length of 2 meters.

$ J'_1 = (40 "kg")/12 dot (4m)^2 = 53.33 "kg" dot m^2 $
$ J'_3 = (160 "kg")/12 dot (16m)^2 = 3413 "kg" dot m^2 $
$ J'_2 = 1/12 20 "kg" dot ((2 m)^2+(2 m)^2) = 13.33 "kg" dot m^2 $




== 3.3 Solving reaction forces and torques

As the accelerations, constraint jacobian and $bold(gamma)$ is found in section 1 it is possible to solve for the Langrange multiplier $bold(lambda)$ using equation 6.3.16:

$ bold(M)bold(dot.double(q)) + bold(Phi_q)^T bold(lambda) = bold(Q)^A <=> bold(lambda) = bold(Phi_q)^T^(-1) (bold(Q)^A - bold(M)bold(dot.double(q))) $

By also using equation 6.6.8 and 6.6.9 the reaction forces and the driver torque can be calculated:

$ bold(F)_i^(''k) = - bold(C)_i^T bold(A)_i^T bold(Phi)^(k^T)_(r_i) bold(lambda)^k $

$ bold(T)_i^(''k) = (bold(s)'^(P^T)_i bold(B)_i^T bold(Phi)^(k^T)_(r_i) - bold(Phi)^(k^T)_(phi.alt_i))bold(lambda)^k $

In this system with the driver at A:
$ bold(T)_1^(''A) = (bold(s)'^(A^T)_1 bold(B)_1^T bold(Phi)^("driver"^T)_(r_1) - bold(Phi)^("driver"^T)_(phi.alt_1))bold(lambda)^"driver" $

Where $bold(s)_1^'^A^T = [-2,0], #v(0.5em) bold(B)_1^T = mat(delim: "[", -sin(phi.alt_1), -cos(phi.alt_1); cos(phi.alt_1), -sin(phi.alt_1)), bold(Phi)^("driver")_(r_1) = [0, 0], bold(Phi)^("driver")_(phi.alt_1) = [1]$

Therefore the $bold(F)_i^(''A) = 0$ as it is multiplied by a zero-vector and $bold(T)_1^(''A) = -bold(lambda)^"driver"$

The driver torque is plotted in the figure below:

#figure(
  image("motor_torque_body1_driver.png")
)



= 4.
In this section the reaction forces of the translation joint on body 2 will be calculated


== 4.1

The equations for the reaction forces and torques are:

$ bold(F)_i^(''k) = - bold(C)_i^T bold(A)_i^T bold(Phi)^(k^T)_(r_i) bold(lambda)^k $

$ bold(T)_i^(''k) = (bold(s)'^(P^T)_i bold(B)_i^T bold(Phi)^(k^T)_(r_i) - bold(Phi)^(k^T)_(phi.alt_i))bold(lambda)^k $

Which is in this system:

$ bold(F)_2^(''C) = - bold(C)_2^T bold(A)_2^T bold(Phi)^("trans"^T)_(r_2) bold(lambda)^"trans" $

$ bold(T)_2^(''C) = (bold(s)'^(C^T)_2 bold(B)_2^T bold(Phi)^("trans"^T)_(r_2) - bold(Phi)^("trans"^T)_(phi.alt_2))bold(lambda)^"trans" $

where $bold(s)_2^C = [0, 1], bold(v)_2 = [1,0], bold(Phi)_(r_2)^("trans") = mat(delim: "[", -bold(v)_2^'^T bold(B)_2^T; 0), bold(Phi)_(phi.alt_2)^("trans") = mat(delim: "[", - bold(v)_2^'^T bold(A)_2^T (bold(r)_3-bold(r)_2)  - bold(v)_2^'^T bold(A)_"23"bold(s)_3^'^"pC"; -bold(v)_2^'^T bold(A)_"23" bold(v)_3^')$

For the reaction forces and torques in the global from the 

The joint reaction forces at the translational joint in body 2 is plotted in the figure below:

#figure(
  image("joint_reactions_body2_frame.png")
)
It can be seen from the plot above that the local x-components are in the order e-12 which is approximately 0 as the computations are numerical.


The joint reaction forces at the translational joint in the world frame is plotted in the figure below:
#figure(
  image("joint_reactions_world_frame.png")
)



