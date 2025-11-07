#set page(header: [
  _Sigurd Mousten Jager Nielsen_ #h(1fr) Aarhus Universitet - Computational Dynamics E25
])

#set page(numbering: "1 of 1")

//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

#figure(
  image("Assignment_description.png")
)


= 1. Setting up a model
In this section the setup will be performed. This section along with 2 and 3 has some overlap with assignment 1 and assignment 2, which has been repeated.

It should be noted that body 1 has not been modelled as distance constraint, but as a separate body to not neglect the dynamics of this body. Therefore the constraint jacobian will not be square, but this is described in section 2, 3 and 4.

== 1.1 Setup of the mechanism
To describe the joints A, B and r are defined:

$ bold(A)(phi.alt) = mat(delim: "[", cos(phi.alt), -sin(phi.alt); sin(phi.alt), cos(phi.alt)) in RR^("2x2"), #h(0.5em) bold(B)(phi.alt) = ded(bold(A), phi.alt) = mat(delim: "[", -sin(phi.alt), -cos(phi.alt); cos(phi.alt), -sin(phi.alt)) in RR^("2x2") $

$ bold(dot(B)) = ded(bold(B), phi.alt)dot(phi.alt) = -bold(A)dot(phi.alt) $

$ bold(r^p = r + A)(phi.alt)bold(s'^P) in RR^2, #h(0.5em) bold(q) = mat(delim: "[", x_1, y_1, phi.alt_1, x_2, y_2, phi.alt_2, x_3, y_3, phi.alt_3)^T in RR^9 $

The points for the joints are described by:

$
s_1^'^"p1" = mat(delim: "[", -frac(L_1, 2); 0),
#h(1em) bold(s_1^'^"p2") = mat(delim: "[", frac(L_1, 2); 0),
#h(1em) bold(s_2^'^"p2") = mat(delim: "[", -frac(L_2, 2); 0),
#h(1em) bold(s_2^'^"p3") = mat(delim: "[", frac(L_2, 2); 0),
#h(1em) bold(s_3^'^"p3") = mat(delim: "[", -frac(L_3, 2); 0),
#h(1em) bold(s_3^'^"p4") = mat(delim: "[", frac(L_3, 2); 0),
 $


*Joints 1 and 4*

The joints between the bodies 1, 3 and ground at point A and D are both absolute position joints and are described by the following equations:

$ bold(Phi^"abs1" (q)) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^1 - C_1))) = mat(delim: "[", 0;0); bold(C_1)=mat(delim: "[", 0;0) $

$ bold(Phi^"abs4" (q)) = mat(delim: "[", bold(r_3 + bold(A"s"_3^'^P^4 - C_4))) = mat(delim: "[", 0;0); bold(C_4)=mat(delim: "[", 20;0) $


The jacobian entries for these joints are:
//CONSTRAINT EQUATIONS 1
$ bold(Phi)^"abs1"_q=
mat(delim: "[", I_"2x2", bold(B)_1 bold("s")_1^'^P^1) $

//CONSTRAINT EQUATIONS 4
$ bold(Phi)^"abs4"_q=
mat(delim: "[", I_"2x2", bold(B)_3 bold("s")_3^'^P^4) $

*Joint 2 and 3*

The joint 2 and 3 between body 1-2 and 2-3 at point 2 and 3 are revolute joints described by the following equations:

$ bold(Phi^"rev2" (q)) = mat(delim: "[", bold(r_1 + bold(A"s"_1^'^P^2)) - (bold(r_2 + bold(A"s"_2^'^P^2))))
 = mat(delim: "[", 0;0) $

$ bold(Phi^"rev3" (q)) = mat(delim: "[", bold(r_2 + bold(A"s"_2^'^P^3)) - (bold(r_3 + bold(A"s"_3^'^P^3))))
 = mat(delim: "[", 0;0) $

The jacobian entries for these joints are:

//CONSTRAINT EQUATIONS 2
$ bold(Phi_q)^"rev2"=
mat(delim: "[", I_"2x2", bold(B)_1 bold("s")_1^'^P^2, -I_"2x2", -bold(B)_2 bold("s")_2^'^P^2) $

//CONSTRAINT EQUATIONS 3
$ bold(Phi_q)^"rev3"=
mat(delim: "[", I_"2x2", bold(B)_2 bold("s")_2^'^P^3, -I_"2x2", -bold(B)_3 bold("s")_3^'^P^3) $


== 1.2 Forward dynamics with Equations of motion using DAE's
By using equation 6.3.18 in Haug's book (2nd ed.) the matrix form of the equation is:


$ mat(delim: "[", 
bold(M), bold(Phi_q)^T;
bold(Phi_q), 0_"8x8"
) mat(delim: "[",
bold(dot.double(q));
bold(lambda)) = mat(delim: "[",
bold(Q)^A;
bold(gamma)
) $

where $bold(M)$ is the positive definite mass matrix of the system, $bold(Phi_q)$ is the constraint jacobian, $bold(lambda)$ is the Lagrange multiplier (vector) for the system, $bold(Q)^A$ is the generalized applied forces and $bold(gamma)$ used in the acceleration equation.

Do note that the matrix for the left hand side has dimensions 17x17 as the jacobian is non-square. This can be seen in the code and will be present in section 2.
$ mat(delim: "[", 
bold(M), bold(Phi_q)^T;
bold(Phi_q), 0_"8x8"
) in RR^(17 times 17) $

for the system the mass matrix will be a diagonal matrix of sizes $3 dot "nb" times 3 dot "nb" = 9 times 9$. For a single body it would be $ mat(delim: "[", m_1, 0, 0;
0, m_1, 0;
0, 0, J^') $
Where $m_1$ and $ J^'$ describe the mass of the element and the local inertia of the body.

For the generalized applied forces: $bold(Q)^A$ the only forces acting on the bodies will be gravity and a torque from the spring-damper acting on body 3 connected to ground:

$ bold(Q)^A = mat(delim: "[", 0, -m_1 dot g, 0, 0, -m_2 dot g, 0, 0, -m_"nb" dot g, -k phi.alt_3 -c dot(phi.alt)_3)^T $

By then numerically integrating $bold(dot.double(q))$ one can obtain $bold(dot(q))$ and $bold(q)$

== 1.3 Baumgarte stabilization
In the forward dynamics simulation Baumgarte stabilization was used to stabilize the mechanism numerically and to enforce the constraint equations as these are not implicitly enforced when integrating $bold(dot.double(q))$.

The baumgarte stabilization used is from equation 7.3.8 in Haug's book:

$ bold(Phi_q dot.double(q)) = bold(gamma) - 2 alpha (bold(Phi_q dot(q))+ bold(Phi)_t) - beta^2bold(Phi) eq.triple hat(gamma) $

= 2 Constraint jacobian $bold(Phi_q)$

There are two constraint jacobians used in this assignment. One is used in the forward dynamics and has the shape $RR^(8 times 9)$. The second is only used initially to get an initial configuration and has the shape $RR^(9 times 9)$ with a constant value of the driver (body 1 left) angle.

== 2.1 Assembling the constraint jacobian $bold(Phi_q)$
The first constraint jacobian can now be assembled using the jacobian entries from each joint:

$ bold(Phi_q) =  mat(delim: "[",
 bold(Phi)^"abs1",0_"2x3",0_"2x3";
 bold(Phi_q)^"rev2", -bold(Phi_q)^"rev2", 0_"2x3";  
 0_"2x3", -bold(Phi_q)^"rev3", bold(Phi_q)^"rev3";  
 0_"2x3",0_"2x3", bold(Phi)^"abs4";
 ) in RR^(8 times 9) $

The second constraint jacobian is:

$ bold(Phi_q) =  mat(delim: "[",
 bold(Phi)^"abs1",0_"2x3",0_"2x3";
 bold(Phi_q)^"rev2", -bold(Phi_q)^"rev2", 0_"2x3";  
 0_"2x3", -bold(Phi_q)^"rev3", bold(Phi_q)^"rev3";  
 0_"2x3",0_"2x3", bold(Phi)^"abs4";
 [0,0,1], 0_"1x3", 0_"1x3";
 ) in RR^(9 times 9) $

== 2.2 Calculating the jacobian
By calling the function assemble_constraints() with the initial conditions the jacobian can be shown for time 0.

#figure(
  image("jacobian.png"),
  caption: "jacobian printout from vs code"
)

= 3 Acceleration equation $bold(gamma)$
The acceleration equations are defined in this section and have been shown in assignment 1, which is why the final $bold(gamma)$ is shown

== 3.1 Assembling the acceleration equations
 The acceleration equations are defined as:

$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)) - 2 bold(Phi)_(bold(q)t)bold(dot(q)) - bold(Phi)_"tt",$
#h(1em) where $bold(dot.double(q)) = mat(delim: "[", dot.double(r); dot.double(phi.alt))$

As $bold(Phi_q)$ does not contain $t$, $bold(Phi)_(bold(q)"t") = [bold(arrow.r(0))]$ and as only nothing is dependant on time:

$ bold(Phi)_"tt" = frac(diff^2bold(Phi), diff^2t) = ded(bold(Phi)_t^D, t) = [arrow.r(0)] $

With the mechanism's equations:

$ bold(gamma) = mat(delim: "[",
dot(phi.alt)_1^2bold(A)_1 bold(s)^('1);
dot(phi.alt)_1^2bold(A)_1 bold(s)^('2)-dot(phi.alt)_2^2bold(A)_2 bold(s)^('2);
dot(phi.alt)_2^2bold(A)_2 bold(s)^('3)-dot(phi.alt)_3^2bold(A)_3 bold(s)^('3);
dot(phi.alt)_3^2bold(A)_3 bold(s)^('4);
) in RR^8 $

The expression for $bold(gamma)$ then becomes:
$bold(gamma) = bold(Phi_q dot.double(q)) = bold(-(Phi_q dot(q))_q dot(q)),$ where $bold(dot.double(q))$ is solved for

Practically one can also solve this using a solver like numpy.linalg.solve which is akin to $backslash$ in matlab, but as forward dynamics are used in this assignment the following equation is solved for instead containing $bold(dot.double(q))$:

$ mat(delim: "[", 
bold(M), bold(Phi_q)^T;
bold(Phi_q), 0_"8x8"
) mat(delim: "[",
bold(dot.double(q));
bold(lambda)) = mat(delim: "[",
bold(Q)^A;
bold(gamma)
) $

= 4 Initial configuration and displacement
The initial configuration is chosen as one wishes. In this assignment the chosen angle is 70 degrees for body 1 (this is body zero in the plot due to index 0).

The initial configuration is then solved for using the Newton Rhapson method as in the other assignments. The second jacobian with the driver is used for this initial configuration and the initial positions are:

$ bold(q)_0 = mat(delim: "[", x_1, y_1, phi.alt_1, x_2, y_2, phi.alt_2, x_3, y_3, phi.alt_3)^T = mat(delim: "[", 1.71, 4.7, 1.22, 16.02, 12.6, 0.25, 24.31, 7.9, 4.21;)^T $

The initial velocities are all set to zero, thus assuming no initial velocity.

The stiffness chosen is $1000000 (n dot m) / "rad"$ and the damping constant is $10000 (n dot m dot s) / "rad"$

The spring has a no stress angle of 270 degrees which equates to body 3 being vertical in the upper configuration.

== 4.1 Plot of initial configuration

The four-bar linkage is displaced to 70 degrees for body 1 and then released. The initial configuration with index 0 naming (used for all plots).

#figure(
  image("initial_config.png"),
  caption: "Initial configuration with numbering"
)

This is animated and by extracting the first and last frame the following two plots are shown:

#figure(
  image("fourbar_frame_first.png"),
  caption: "Initial configuration from animation"
)

#figure(
  image("fourbar_frame_last.png"),
  caption: "Steady state of from last frame of the animation"
)

= 5 Plots of all generalized coordinates
Note that the notation in the plots are index 0 based so that the bodies are called 0, 1, 2.

@pos_1, @pos_2 and @pos_3 show the positions of body 1-3 over time as generalized coordinates.
#figure(
  image("pos_body_0.png"),
  caption: "Position of body 1 in global coordinates"
)<pos_1>

#figure(
  image("pos_body_1.png"),
  caption: "Position of body 2 in global coordinates"
)<pos_2>

#figure(
  image("pos_body_2.png"),
  caption: "Position of body 3 in global coordinates"
)<pos_3>

@vel_acc_1, @vel_acc_2 and @vel_acc_3 show the velocities and accelerations of body 1-3 over time as generalized coordinates.
#figure(
  image("vel_acc_body_0.png"),
  caption: "Velocity and acceleration of body 1 in global coordinates"
)<vel_acc_1>

#figure(
  image("vel_acc_body_1.png"),
  caption: "Velocity and acceleration of body 2 in global coordinates"
)<vel_acc_2>

#figure(
  image("vel_acc_body_2.png"),
  caption: "Velocity and acceleration of body 3 in global coordinates"
)<vel_acc_3>

= 6 Plots of spring-damper forces vs time

The spring-damper forces are plotted in @spring_damper and shows how there are large moments due to the rotational spring-damper connected to body 3 from ground and how it stabilizes around 2-2.5 seconds.

#figure(
  image("spring_damper_body_2.png"),
  caption: "Torque of the rotational spring, damper and summed torques vs time"
)<spring_damper>

= 7 Plots of the reaction forces in joint 3 vs time
Here the plots of the reaction forces are shown over time. the plots are in both local and global reference frames. To get the reaction forces the general equation 6.6.8 in Haug's book is used:

$ bold(F)_i^(''k) = - bold(C)_i^T bold(A)_i^T bold(Phi)^(k^T)_(r_i) bold(lambda)^k $

@reaction_forces shows the reaction forces and it should be noted that some of the forces are about 25000 N close to the initial configuration.
#figure(
  image("reaction_forces_body_2.png"),
  caption: "Reaction forces of body 2 and 3 in global and local coordinates"
)<reaction_forces> 


= 8 Additional plots
As with the large forces of the spring-damper stabilization was needed and implemented with an alpha value of 10 and a beta value of 200. Apparently from Haug's book there is no general and uniformly valid way of choosing the alpha and beta values at his time of writing.
#figure(
  image("constraint_body_2.png"),
  caption: "Constraint residual plot"
) <stabilization_plot>

If the values of alpha and beta are reduced to 1 and 20 respectively one can observe joint "dislocation" in the movement of the mechanism at the timepoints correlating to the peak in @stabilization_plot although this figure is with alpha = 10 and beta = 200 the same trend is observed.

As the beta term is squared and multiplied by $bold(Phi)$ this could enforce the positional correction and higher values here might stabilize better.

For this assignment the explicit RK45 (Runge-Kutta) integration scheme was used.  

The choice of integration scheme, stabilization coefficients and step size should probably be varied to get a more accurate result and optimize solving time.