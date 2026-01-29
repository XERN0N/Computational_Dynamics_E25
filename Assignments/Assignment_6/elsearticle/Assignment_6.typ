#import "@preview/elsearticle:1.0.0": *
#import "@preview/mannot:0.3.1": *

//#set page(numbering: "1 of 1")
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#show: elsearticle.with(
  title: "Computational Dynamics Q assignment 3",
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
This report covers assignment Q3 in Computational Dynamics E25 at Aarhus University. There is an accompanying jupyter notebook in python with the code. Some of the theory is omitted in this assignment 3 as it has already been covered in assignment 1-2 and some theory will be repeated.


= ATBI algorithm
The ATBI algorithm consists of two loops, one a scatter-loop and the other a gather-loop.

== Scatter kinematic loop

The scatter loop recursively runs through all the bodies from base to tip and uses the body level kinematics to get the kinematics for the space spaceshuttle. The spaceshuttle (base) is unlike q2 not defined as being stationary in the assignment and thus $cal(V)_"base" eq.not 0, #h(1em) alpha_"base" eq.not 0$

a hinge map for the "free" joint is introduced as of fig. 3.2 in #cite(<Compdyn_abhi>):

//HINGE MAP
$
  H = mat(delim: "[", bold(I)_6) = 
  mat(delim: "[",
  1, 0, 0, 0, 0, 0;
  0, 1, 0, 0, 0, 0;
  0, 0, 1, 0, 0, 0;
  0, 0, 0, 1, 0, 0;
  0, 0, 0, 0, 1, 0;
  0, 0, 0, 0, 0, 1;
  )
$
This hingemap is in effect just an identity map and the matrix multiplication can be omitted, where the hinge map used.

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

== Coriolis spatial acceleration

$frak(a)$ is the coriolis spatial acceleration from eq. 5.9 #cite(<Compdyn_abhi>) which in this case as there are both rotational joints and prismatic joints reduces to eq. 5.11 in the case of body frame derivatives: 
$
  frak(a) (k) = tilde(cal(V))(k)Delta_cal(V)(k) - overline(Delta)_cal(V)(k)Delta_cal(V)(k) + frac(d_(k+1) H^(T)(k),d t)dot(theta)(k)
$

It should be noted that for the $phi.alt$ the length is no longer constant in the case of the free joint or decoupling of the spaceshuttle to the inertia frame.
The accelerations can then be obtained for the spaceshuttle.

== Gather loop

For the forward dynamics the gather loop recursively iterates from tip to base of the spaceshuttle to solve for the generalized accelerations corresponding to the prescribed generalized forces and initial conditions.

The articulated body inertia (ATBI) is used during the gather-loop and the algorithm can be found as algorithm 7.1. #cite(<Compdyn_abhi>)

First the inertias are calculated using:

#nonumeq(
$
  P(k) = phi.alt(k, k-1)P^+(k) phi.alt^T (k, k-1) + M(k)
$
)
Then the hinge inertia $D$ and kalman gain $cal(G)$ can be calculated.

#nonumeq(
$
  D(k) = H(k)P(k)H^T (k), #h(1em) cal(G)(k) = P(k)H^T (k)D^(-1)(k)
$
)

Then the projection operator $overline(tau)$.

#nonumeq(
$
  overline(tau)(k) = bold(I) - cal(G)(k)H(k)
$
)

and updating the projected articulated inertia $P^+$.

#nonumeq(
$
  P^+ (k) = overline(tau)(k)P(k)
$
)

After the update the residual spatial force $frak(z)$

#nonumeq(
$
  frak(z)(k) = phi.alt(k, k-1)frak(z^+)(k-1)+P(k)frak(a)(k) + frak(b)(k)
$
)

Then the ATBI innovations generalized force $epsilon.alt$ and ATBI innovations generalized acceleration $nu$

#nonumeq(
$
  epsilon.alt(k) = cal(T)(k) - H(k)frak(z)(k), #h(1em) nu = D^(-1)(k)epsilon.alt(k)
$
)

And finally the residual spatial force can be updated.

#nonumeq(
$
  frak(z^+)(k) = frak(z)(k) + cal(G)(k)epsilon.alt(k)
$
)

As described in assignment 2, the gyroscopic force vector for the body frame $frak(b)(k) = overline(cal(V))(k)M(k)V(k)$ is used.
The $overline(cal(V))(z)$ is the spatial vector cross-product in eq. 1.25 #cite(<Compdyn_abhi>) of the $cal(V)$ spatial velocities:
$
  overline(z) eq.delta mat(delim: "[",
  tilde(x), tilde(y);
  bold(0)_3, tilde(x)
  ) in bb(R)^(6 times 6)
$ 

$M(k)$ denotes the spatial inertia from eq. 2.7 #cite(<Compdyn_abhi>):
$
M(z) = mat(delim: "[",
  cal(J)(z), m tilde(p)(z); -m tilde(p)(z), m bold(I)_3
) in bb(R)^(6 times 6)
$
with eq. 2.11 #cite(<Compdyn_abhi>): $cal(J)(z) = cal(J)(c) -m tilde(p)(z)tilde(p)(z)$ and $m$ being the mass of the body and $p = attach(frak(l), tl: bb(B))(z,C)$ describing the distance from $z$ to $C$.

== Scatter spatial and generalized accelerations loop

A scatter-loop is used to get the spatial and generalized velocities from the forces applied in the gather-loop.

The spatial accelerations $alpha$ 
#nonumeq(
$
  alpha^+(k) = phi.alt^T (k+1, k)alpha(k+1)
$
)

The generalized accelerations $dot.double(theta)$
#nonumeq(
$
  dot.double(theta)(k) = nu(k) - cal(G)^T (k)alpha^+ (k)
$
)

and $alpha$ is updated for the next iteration.
#nonumeq(
$
  alpha(k) = alpha^+ (k) + H^T dot.double(theta)(k) + frak(a)(k)
$
)

= Equation of motion
The equations of motion for the spaceshuttle are defined using the states as hinted in the assignment description.

As this is a forward dynamics problem the generalized forces are applied to the bodies with initial conditions. Then using an Initial Value Problem integrator (Solve_IVP in SciPy) a function mapping the state to the state derivative $S arrow dot(S)$ can be integrated using numerical integration methods. The state and state derivative for the system in question is defined as:

$
  S eq.delta mat(delim: "[",
  attach(underline(q), tl: 6, br: 5);
  attach(l, tl: 6, br: 5);
  attach(underline(q), tl: 5, br: 4);
  attach(theta, tl: 4, br: 3);
  attach(theta, tl: 3, br: 2);
  attach(underline(q), tl: 2, br: 1);
  attach(omega, tl: 5, br: 4);
  attach(beta, tl: 4, br: 3);
  attach(beta, tl: 3, br: 2);
  attach(omega, tl: 2, br: 1);
  )
  in bb(R)^(31),

  #h(1em)

  dot(S) eq.delta mat(delim: "[",
  attach(dot(underline(q)), tl: 6, br: 5);
  attach(dot(l), tl: 6, br: 5);
  attach(dot(underline(q)), tl: 5, br: 4);
  attach(beta, tl: 4, br: 3);
  attach(beta, tl: 3, br: 2);
  attach(dot(underline(q)), tl: 2, br: 1);
  attach(gamma, tl: 5, br: 4);
  attach(gamma, tl: 4, br: 3);
  attach(gamma, tl: 3, br: 2);
  attach(gamma, tl: 2, br: 1);
  )
  in bb(R)^(31)
$

Where $underline(q), theta$ are the generalized orientations, $l$ is the generalized position, $beta, omega$ are generalized velocities, the generalized accelerations are $gamma in bb(R)^3 or bb(R)$ depending on the joint DoF. 

The $dot(underline(q))$ is the quaternion rate calculated using eq. B.34 #cite(<Compdyn_abhi>) as seen in #ref(<q_dot>): 

//Quaternion derivatives
$
  attach(dot(underline(q))_bb(B), tl: bb(I)) = 1/2 mat(delim: "[",
  -attach(tilde(w), tl: bb(B)), attach(tilde(w), tl: bb(B));
  attach(w, tl: bb(B))^T, 0;
  )underline(q)
$<q_dot>

In code the quaternions are normalized to ensure unit quaternions when integrating.

The spaceshuttle has three different joint types, where purely rotational joints are spherical joints and revolute joints. The spherical joints have 3 rotational DoF and 0 translational DoF, whereas the revolute joints have 1 rotational DoF and 0 translational DoF. The generalized orientation and velocity of a revolute joint is constrained to the 1 rotational DoF and the orientation of this type of joint is given by eq. B.12 #cite(<Compdyn_abhi>) as #ref(<euler_params>):

$
  q_0 eq.delta cos(theta/2), #h(1em) q eq.delta sin(theta/2) bold(frak(n))
$<euler_params>

For the revolute joints on the spaceshuttle the x-axis are collinear and thus for these two joints $bold(frak(n)) = mat(delim: "[", 1; 0; 0)$ which makes #ref(<euler_params>) reduce to #ref(<revolute_quaternion>):

$
  attach(underline(q), tl: bb(I), br: bb(B)) = mat(delim: "[",
  sin(theta/2); 0; 0; cos(theta/2)
  )
$<revolute_quaternion>

For the free joint, where linear movement can happen, there are some difficulties when differentiating the linear position as the positions are stored in a body frame, but as in the book in eq. 1.12 #cite(<Compdyn_abhi>) it has to be differentiated in a non-rotating reference frame which is why the linear velocity is rotated into the inertial reference frame.

$
  dot(p) = v - omega times p
$


= Simulation
The results of the simulations can be seen in figure #ref(<wrench>) which shows the spatial accelerations for body 1 and 5 for the spaceshuttle. The forces and moments are the ones acting on the spaceshuttle as per the csv file in the assignment. This result is the same as the plot from the assignment question 3.3.


#figure(
  image("images/wrench_result.png"),
   caption: "Spatial accelerations of the bodies 1 and 5 on the spaceshuttle"
)<wrench>


#bibliography("refs.bib")