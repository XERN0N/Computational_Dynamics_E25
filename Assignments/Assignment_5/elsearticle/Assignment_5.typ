#import "@preview/elsearticle:1.0.0": *
#import "@preview/mannot:0.3.1": *

//#set page(numbering: "1 of 1")
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#show: elsearticle.with(
  title: "Computational Dynamics Q assignment 2",
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
This report covers assignment Q2 in Computational Dynamics E25 at Aarhus University. There is an accompanying jupyter notebook in python with the code. Some of the theory is omitted in this assignment 2 as it has already been covered in assignment 1 and some theory will be repeated.


= Equation of motion
The equations of motion for the spaceshuttle are defined using the states as hinted in the assignment description.

As this is an inverse dynamics problem the motion is prescribed using accelerations and initial conditions. Then using an Initial Value Problem integrator (Solve_IVP in SciPy) a function mapping the state to the state derivative $S arrow dot(S)$ can be integrated using numerical integration methods. The state and state derivative for the system in question is defined as:

$
  S eq.delta mat(delim: "[",
  attach(underline(q), tl: 5, br: 4);
  attach(theta, tl: 4, br: 3);
  attach(theta, tl: 3, br: 2);
  attach(underline(q), tl: 2, br: 1);
  attach(omega, tl: 5, br: 4);
  attach(beta, tl: 4, br: 3);
  attach(beta, tl: 3, br: 2);
  attach(omega, tl: 2, br: 1);
  )
  in bb(R)^(18),

  #h(1em)

    dot(S) eq.delta mat(delim: "[",
  attach(dot(underline(q)), tl: 5, br: 4);
  attach(beta, tl: 4, br: 3);
  attach(beta, tl: 3, br: 2);
  attach(dot(underline(q)), tl: 2, br: 1);
  attach(gamma, tl: 5, br: 4);
  attach(gamma, tl: 4, br: 3);
  attach(gamma, tl: 3, br: 2);
  attach(gamma, tl: 2, br: 1);
  )
  in bb(R)^(18)
$

Where $underline(q), theta$ are the generalized orientations, $beta, omega$ are generalized velocities, the generalized accelerations are $gamma in bb(R)^3 or bb(R)$ depending on the joint DoF. 

The $dot(underline(q))$ is the quaternion rate calculated using eq. B.34 #cite(<Compdyn_abhi>) as seen in #ref(<q_dot>): 

//Quaternion derivatives
$
  attach(dot(underline(q))_bb(B), tl: bb(I)) = 1/2 mat(delim: "[",
  -attach(tilde(w), tl: bb(B)), attach(tilde(w), tl: bb(B));
  attach(tilde(w), tl: bb(B))^T, 0;
  )underline(q)
$<q_dot>

In code the quaternions are normalized to ensure unit quaternions when integrating.

The spaceshuttle has two different joint types, namely spherical joints and revolute joints. The spherical joints have 3 rotational DoF and 0 translational DoF, whereas the revolute joints have 1 rotational DoF and 0 translational DoF. The generalized orientation and velocity of a revolute joint is constrained to the 1 rotational DoF and the orientation of this type of joint is given by eq. B.12 #cite(<Compdyn_abhi>) as #ref(<euler_params>):

$
  q_0 eq.delta cos(theta/2), #h(1em) q eq.delta sin(theta/2) bold(frak(n))
$<euler_params>

For the revolute joints on the spaceshuttle the x-axis are collinear and thus for these two joints $bold(frak(n)) = mat(delim: "[", 1; 0; 0)$ which makes #ref(<euler_params>) reduce to #ref(<revolute_quaternion>):

$
  attach(underline(q), tl: bb(I), br: bb(B)) = mat(delim: "[",
  sin(theta/2); 0; 0; cos(theta/2)
  )
$<revolute_quaternion>

= Scatter loop

The scatter loop recursively runs through all the bodies from base to tip and uses the body level kinematics to get the velocities and accelerations for the space spaceshuttle. The spaceshuttle (base) is defined as being stationary in the assignment and thus $cal(V)_"base" = 0, #h(1em) alpha_"base" = 0$

a hinge map for the spherical joint is introduced as of fig. 3.2 in #cite(<Compdyn_abhi>):

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

The propagation of accelerations can be done using eq. 5.8 #cite(<Compdyn_abhi>) as #ref(<rel_acc>):

$
  alpha (k)= phi.alt^T (k+1, k) alpha (k+1) + Delta_alpha (k) + frak(a)(k)
$<rel_acc>

Where $Delta_alpha (k) = H^T gamma (k)$ and $frak(a)$ is the coriolis spatial acceleration from eq. 5.9 #cite(<Compdyn_abhi>) which in this case as there are only purely rotational joints reduces to eq. 5.15 (and 5.16): 
$
  frak(a) (k) = tilde(cal(V))(k)Delta_cal(V) (k)
$

then the kinematics can be obtained for the spaceshuttle.

= Gather loop

For the inverse dynamics the gather loop recursively iterates from tip to base of the spaceshuttle to obtain the dynamics corresponding to the prescribed motion and initial conditions.

To get the spatial forces for each body the single rigid body equations of motion are used in eq. 5.4 #cite(<Compdyn_abhi>) as #ref(<gather>):

$
  frak(f)(k) = phi.alt(k,k-1)frak(f)(k,k-1) + M(k)alpha(k) + frak(b)(k)
$<gather>

Where the gyroscopic force vector for the body frame $frak(b)(k) = overline(cal(V))(k)M(k)V(k)$ is used.
The $overline(cal(V))(z)$ is the spatial vector cross-product in eq. 1.25 #cite(<Compdyn_abhi>) of the $cal(V)$ spatial velocities:
$
  overline(z) eq.delta mat(delim: "[",
  tilde(x), tilde(y); bold(0)_3, tilde(x)
  ) in bb(R)^(6 times 6)
$ 

$M(k)$ denotes the spatial inertia from eq. 2.7 #cite(<Compdyn_abhi>):
$
M(z) = mat(delim: "[",
  cal(J)(z), m tilde(p)(z); -m tilde(p)(z), m bold(I)_3
) in bb(R)^(6 times 6)
$
with eq. 2.11 #cite(<Compdyn_abhi>): $cal(J)(z) = cal(J)(c) -m tilde(p)(z)tilde(p)(z)$ and $m$ being the mass of the body and $p = attach(frak(l), tl: bb(B))(z,C)$ describing the distance from $z$ to $C$.


= Simulation
The results of the simulations can be seen in figure #ref(<wrench>) which shows the required forces and moments to get the prescribed motion of the spaceshuttle. The forces and moments are the ones acting on the spaceshuttle from the arm and payload. This result is the same as the plot from the assignment question 2.4. In the code the gather loop output at k=5 gives these exact forces and moments as the centroid and hinge is coincident.


#figure(
  image("images/wrench_result.png"),
   caption: "Wrench on spaceshuttle from arm"
)<wrench>


#bibliography("refs.bib")