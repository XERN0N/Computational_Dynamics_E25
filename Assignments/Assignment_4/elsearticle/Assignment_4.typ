#import "@preview/elsearticle:1.0.0": *
#import "@preview/mannot:0.3.1": *

//#set page(numbering: "1 of 1")
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#show: elsearticle.with(
  title: "Computational Dynamics Q assignment 1",
  authors: (
    (
      name: "Sigurd Mousten Jager Nielsen",
      affiliation: "Aarhus Universitet",
      corr: "202108107@uni.au.dk",
      id: none,
    ),
  ),
  journal: none,
  abstract: none,
  keywords: none,
  format: "preprint",
  // line-numbering: true,
)

//AU-LOGO
#figure(
  image("images/au_logo.png", width: 25%),
)
//PAGEBREAK
#colbreak()

#outline(title: "Table of contents")

#colbreak()

= Introduction
This report covers assignment Q1 in Computational Dynamics E25 at Aarhus University. There is an accompanying jupyter notebook in python with the code.

/*
The following is given in the assignment:

#nonumeq(
$
 attach(l, tl: bb(B)_1)(w, x) = [1.0, -0.1, 0.4]^T, #h(1em) attach(l, tl: bb(B)_2)(w, x) = [1.0, 0, 0.2]^T 
$
)

//TODO
#nonumeq(
$
 attach(q, tl: bb(I))(w, x) = [1.0, -0.1, 0.4]^T, #h(1em) attach(l, tl: bb(B)_2)(w, x) = [1.0, 0, 0.2]^T 
$
)*/


= Single transformation

To get the rotation matrix from the quaternions requires usage of eq. B.14 #cite(<Compdyn_abhi>)


//quaternion convertion
$
  attach(frak(R), tl: bb(F), br: bb(G))(attach(underline(q), tl: bb(F), br: bb(G))) = 
  (q_0 bold(I)_3 + tilde(q))^2 + q q^T
$

The homogenious transform is described by eq. 1.3 #cite(<Compdyn_abhi>)

$
  attach(bb(T), tl: bb(F), br: bb(G)) eq.delta mat(delim: "[",
  attach(frak(R), tl: bb(F), br: bb(G)), attach(frak(l), tl: bb(F))(bb(F),bb(G));
  0, 1
  )
$

With $attach(frak(l), tl: bb(F))(bb(F),bb(G))$ being the length from $bb(F)$ to $bb(G)$ as seen from $bb(F)$.

The position of a point in one coordinate frame can be expressed in another coordinate frame by using eq. 1.4 #cite(<Compdyn_abhi>):

$
 mat(delim: "[", attach(frak(p), tl: bb(F), )(bb(F),bb(O)); 1) = attach(bb(T), tl: bb(F), br: bb(G)) mat(delim: "[", attach(frak(p), tl: bb(G), )(bb(G),bb(O)); 1)
$

In the current case eq. 1.2 #cite(<Compdyn_abhi>) will be used instead to represent the vector from the point $w$ to $z$ in the reference frame $bb(B)_1$ as the points $x$ and $y$ are coincident:


//Representing wz by rotating yz onto body 1
$
  attach(frak(l), tl: bb(B)_1)(w,z) = attach(frak(l), tl: bb(B)_1)(w, x) + attach(frak(R), tl: bb(B)_1, br: bb(B)_2) attach(frak(l), tl: bb(B)_2)(y, z)
$<l_in_b1>

A skew-symmetric matrix $tilde(l)$ from a vector $l$ is defined from eq. 1.9 #cite(<Compdyn_abhi>):

$
  tilde(l) eq.delta mat(delim: "[", 
  0, -c, b;
  c, 0, -a;
  -b, a, 0
  ),
  #h(1em)
  l eq.delta mat(delim: "[", a; b; c)
$<skew_sym>

The rigid body transform from eq. 1.29 and 1.31 is #cite(<Compdyn_abhi>):

$
  phi.alt(x,y) = 
  mat(delim: "[", 
   bold(I)_3, attach(tilde(l)(x,y), tl: bb(F));
    bold(0)_3,  bold(I)_3
  )   mat(delim: "[", 
   attach(frak(R), tl: bb(F), br: bb(G)), bold(0)_3;
    bold(0)_3, attach(frak(R), tl: bb(F), br: bb(G))
  )
$<rb_transform>

To then get the velocities in $attach(cal(V), tl: bb(I))(bb(I),z)$ from $attach(cal(V), tl: bb(B)_1)(bb(I),w)$ eq. @skew_sym and @rb_transform are used with eq. 1.33 #cite(<Compdyn_abhi>):

//RB STANDARD
$
  attach(cal(V), tl: bb(F))(y) = phi.alt(x,y)^T attach(cal(V), tl: bb(G))(x)
$<rb_standard>

Which is used with eq. @l_in_b1 to transform from point $w$ to $z$: 
$
  attach(cal(V), tl: bb(B)_1)(bb(I),z) = phi.alt(w,z)^T attach(cal(V), tl: bb(B)_1)(bb(I),w) = 
  mat(delim: "[", 
   bold(I)_3, bold(0)_3;
    -attach(tilde(l)(w,z), tl: bb(B)_1),  bold(I)_3
  )
  attach(cal(V), tl: bb(B)_1)(bb(I),w)
$<rb_vel>

As a last step the velocity is rotated from $bb(B)_1$ to match the inertial fixed reference frame $bb(I)$ which due to only a change in orientation reduces eq. @rb_standard to:

//RESULT Q 1.1.1
$
  attach(cal(V), tl: bb(I))(bb(I),z) = 
  mat(delim: "[", attach(frak(R), tl: bb(bb(I)), br: bb(B)_1), bold(0)_3;
  bold(0)_3, attach(frak(R), tl: bb(bb(I)), br: bb(B)_1)) attach(cal(V), tl: bb(B)_1)(bb(I),z)
  approx #h(1em) markrect(mat(delim: "[", 0.567; 1.267; 0.067; 1.973; -0.567; -0.607), outset: #0.5em)
$<rot_to_I>

= Chaining

With the connection between body $bb(B)_1$ and $bb(B)_2$ now changed from a rigid connection to a spherical hinge, a hinge map  is introduced as of fig. 3.2 in #cite(<Compdyn_abhi>):

//HINGE MAP
$
  H = mat(delim: "[", bold(I)_3; bold(0)_3)^T = 
  mat(delim: "[",
  0, 0, 1, 0, 0, 0;
  0, 1, 0, 0, 0, 0;
  1, 0, 0, 0, 0, 0;
  )
$

Then by using eq. 3.7 #cite(<Compdyn_abhi>) the relative spatial velocity $Delta_cal(V)$ governing the two bodies is:

//DELTA V
$
  Delta_cal(V) (k) = H^T (k) beta (k)
$

Where $beta(k)$ denotes the generalized velocity coordinates vector across the body. Then by using eq. 3.19b #cite(<Compdyn_abhi>) the velocity of a body  $k$ can be obtained:
//Relative velocity
$
  cal(V) (k)= phi.alt^T (k+1, k) cal(V) (k+1) + Delta_cal(V) (k)
$<rel_vel>

Which in this case becomes:
//Relative velocity body 1-2
$
  cal(V) (bb(B)_2)= phi.alt^T (bb(B)_1, bb(B)_2) cal(V) (bb(B)_1) + Delta_cal(V) (bb(B)_2)
$<body_2_rel>

So for the case in Q 1.1.2 the relative angular velocity $attach(w, tl: bb(B)_1)(bb(B)_1, bb(B)_2)$ is given and a series of transformations from the spatial velocities of the point $w$ in $bb(B)_1$ are needed. They are the following and start with the transformation from point $w$ to $x$ in $bb(B)_1$:

//First
$
  attach(cal(V), tl: bb(B)_1)(bb(I),x) = phi.alt(w,x)^T attach(cal(V), tl: bb(B)_1)(bb(I),w) 
$
From $x$ the relative velocities of the two bodies are added as they are expressed in $bb(B)_1$:

//Second
$
    attach(cal(V), tl: bb(B)_1)(bb(I),x+Delta_cal(V)(bb(B)_2)) = attach(cal(V), tl: bb(B)_1)(bb(I),w) + attach(Delta_cal(V), tl: bb(B)_1) (bb(B)_2)
$

This is followed by a transformation with only a difference in rotation between $bb(B)_1$ and $bb(B)_2$
//Third
$
  attach(cal(V), tl: bb(B)_2)(bb(I),y) = phi.alt(x,y)^T attach(cal(V), tl: bb(B)_1)(bb(I),x+Delta_cal(V)(bb(B)_2))
$

Then a transformation from $y$ to $z$ which is only displaced by a length.
//Fourth
$
  attach(cal(V), tl: bb(B)_2)(bb(I),z) = phi.alt(y,z)^T attach(cal(V), tl: bb(B)_2)(bb(I),y) 
$

After the velocities are propagated to the outermost body, the velocities can then be expressed in the inertial fixed reference frame $bb(I)$ using the same rotation as in eq. @rot_to_I.

//Solution to Q1.1.2
$
  attach(cal(V), tl: bb(I))(bb(I), z) = 
  mat(delim: "[", attach(frak(R), tl: bb(bb(I)), br: bb(B)_2), bold(0)_3;
  bold(0)_3, attach(frak(R), tl: bb(bb(I)), br: bb(B)_2)) attach(cal(V), tl: bb(B)_2)(bb(I),z) approx #h(1em) markrect(mat(delim: "[", 3.5; 3; -0.2; 2.853; -2.167; -1.327), outset: #0.5em)
$<sol_112>

= Equations of motion
The equations of motion (EoM) for the body is governed by eq. 2.28 #cite(<Compdyn_abhi>) for a rigid body with forces and moments about an arbitrary point $z$ in the body reference frame:

//EOM
$
  frak(f)(z) = M(z)dot(beta)(z) + frak(b)(z), #h(1em) frak(b)(z) = overline(cal(V))(z)M(z) cal(V)(z)
$
With $frak(f)(z) = mat(delim: "[", N(z); F(z)) in bb(R)^6$ denoting the spatial forces. and the spatial inertia from eq. 2.7 #cite(<Compdyn_abhi>):
$
M(z) = mat(delim: "[",
  cal(J)(z), m tilde(p)(z); -m tilde(p)(z), m bold(I)_3
) in bb(R)^(6 times 6)
$
with eq. 2.11 #cite(<Compdyn_abhi>): $cal(J)(z) = cal(J)(c) -m tilde(p)(z)tilde(p)(z)$ and $m$ being the mass of the body and $p = attach(frak(l), tl: bb(B))(z,C)$ describing the distance from $z$ to $C$.

The $overline(cal(V))(z)$ is the spatial vector cross-product in eq. 1.25 #cite(<Compdyn_abhi>) of the $cal(V)$ spatial velocities:
$
  overline(z) eq.delta mat(delim: "[",
  tilde(x), tilde(y); bold(0)_3, tilde(x)
  ) in bb(R)^(6 times 6)
$ 

As it is needed for the simulation to get the generalized accelerations $dot(beta)$ for each timestep, $dot(beta)$ is solved for:

//EOM BETA_dot
$
  dot(beta)(z) = M(z)^(-1) (frak(f)(z) - frak(b)(z)) = M(z)^(-1) (frak(f)(z) - overline(cal(V))(z)M(z) cal(V)(z)) 
$

As $M(z)^(-1)$ is not dependent on time it can be precomputed and as such imposes a constant $cal(O)(1)$ computational cost and not $cal(O)(N)$ for $N$ being the number of timesteps in the simulation.

It is also needed to get the time derivative of quaternions in eq. B.34 #cite(<Compdyn_abhi>) when integrating over time in section @simulation:


//Quaternion derivatives
$
  attach(dot(underline(q))_bb(B), tl: bb(I)) = 1/2 mat(delim: "[",
  -attach(tilde(w), tl: bb(B)), attach(tilde(w), tl: bb(B));
  attach(tilde(w), tl: bb(B))^T, 0;
  )underline(q)
$<q_dot>

The linear velocity to the body from the Inertial fixed reference frame $attach(dot(frak(l))_bb(B) (bb(I), z), tl: bb(I))$ is:

$
    attach(dot(frak(l)) (bb(I), z), tl: bb(I)) = attach(frak(R), tl:bb(I), br: bb(B))attach(nu, tl: bb(B))
$

The choice of generalized velocities is $attach(frak(l) (bb(I), z), tl: bb(I))$ which is the linear position from $bb(I)$ to $z$ as seen from $bb(I)$.

The state vector can then be assembled:

//STATE
$
  s = mat(delim: "[",
  attach(underline(q)_bb(B), tl: bb(I));
  attach(frak(l) (bb(I), z), tl: bb(I));
  beta_bb(B);
  )
$

Then the derivative of the state with respect to time is obtained:
//STATE DERIVATIVE
$
  dot(s) = mat(delim: "[",
  attach(dot(underline(q))_bb(B), tl: bb(I));
  attach(dot(frak(l)) (bb(I), z), tl: bb(I));
  attach(dot(beta)_bb(B));
  ) = mat(delim: "[",
  1/2 mat(delim: "[",
  -attach(tilde(w), tl: bb(B)), attach(tilde(w), tl: bb(B));
  attach(tilde(w), tl: bb(B))^T, 0;
  )underline(q);
  attach(frak(R), tl:bb(I), br: bb(B))attach(nu, tl: bb(B));
  M(z)^(-1) (frak(f)(z) - overline(cal(V))(z)M(z) cal(V)(z))
  )
$

The equations of motions for the body is described within the state $s$ with the derivative being $dot(s)$.

= Simulation <simulation>
To simulate the system the states $s$ and derivative $dot(s)$ is integrated using numerical integration using a 4th order Runge-Kutta integration scheme ("RK45") for 5 seconds with the relative and absolute tolerances of $1 dot 10^(-9)$ with a variable step solver using the Solve IVP method from SciPy and evaluated at 100 equidistant points. The initial conditions are given and with the bodyframe $C$ coinciding and aligned with $bb(I)$:

$
  s_0 = mat(delim: "[", attach(underline(q)_bb(B), tl: bb(I))^T,
  attach(frak(l) (bb(I), z), tl: bb(I))^T,
  beta_bb(B)^T)^T= mat(delim: "[", 0, 0, 0, 1, 0.5, 1, 0, 0, 0, 0, 0, 0, 0,)^T
$

The simulation results can be seen in Figures @2d-plot and @3d-plot below:


#figure(
  image("images/2d_trajectories.png"),
  caption: "2D plots of the trajectories of C and z viewed from the x-y (right) and x-z (left) planes with respect to the inertial frame"
)<2d-plot>


#figure(
  image("images/3d_trajectory.png"),
  caption: "3D plot of the trajectories of C and z as seen by I"
)<3d-plot>

#bibliography("refs.bib")