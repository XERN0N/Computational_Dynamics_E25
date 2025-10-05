#set page(header: [
  _Sigurd Mousten Jager Nielsen_ #h(1fr) Aarhus Universitet - Computational Dynamics E25
])

#set page(numbering: "1 of 1")

//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)


#figure(
  image("image.png"))

= 1.

There are 3 bodies in the mechanism with each 3 DoF. There are also 4 joints: 3 revolute joints and 1 translational joint. There is also 1 driver with constant rotational speed.

$"DoF" = "nb" dot 3 - "nc" = 3 dot 3 - (4 dot 2 + 1 dot 1) = 0$




= 2.




