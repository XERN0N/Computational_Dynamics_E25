import jax
import jax.numpy as jnp
from mech_types import Mechanism
#from jax import jit, vmap, jacfwd, grad
#import numpy as np
#import matplotlib.pyplot as plt
#import timeit

jax.config.update('jax_enable_x64', True)

class MechanismBuilder:

    def __init__(self, instance_name:str, world_pose=(0, 0, 0), gravity=(0, -9.81)):
        self.mechanism_name = instance_name
        self.world = {"pose": world_pose, "points": {"O": (0.0, 0.0)}, "frames": {}}
        self.bodies = []
        self.joints = []
        self.drivers = []
        self.options = {"gravity": gravity}        

    def add_world_point(self, name:str, ground_point:list[float]|tuple[float, float]) -> None:
        x, y = ground_point
        self.world["points"][name] = tuple((float(x), float(y)))
        print(f"The point has been added and can be viewed with expose_dict(\"world\")")

    def add_body(
        self,
        body_name: str,
        point_dict: dict[str, tuple[float, float]],
        initial_configuration: tuple[float, float, float],
        mass: float | int | str,
        inertia_zz: float | int | str,
        dynamic: bool = True,
    ) -> None:

        initial_configuration = tuple(map(float, initial_configuration))
        mass = float(mass)
        inertia_zz = float(inertia_zz)

        self.bodies.append(
            {
                "name": body_name,
                "points": point_dict,
                "initial_configuration": initial_configuration,
                "mass": mass,
                "inertia_zz": inertia_zz,
                "dynamic": dynamic,
            }
        )

        print(f"The body has been added and can be viewed with expose_dict(\"bodies\")")

    def add_joint(self, name: str, joint_type: str, endpoint_1: dict, endpoint_2: dict, constraint_arguments: dict) -> None:
        jt = joint_type.lower().strip()
        joint_types = ("revolute", "distance", "translation", "prismatic", "angle", "gear", "composite", "weld")
        if jt not in joint_types:
            raise ValueError(f"Unknown joint type: {joint_type}")
        self.joints.append(
            {
                "name": name,
                "joint_type": jt,
                "endpoint_1": endpoint_1,
                "endpoint_2": endpoint_2,
                "constraint_arguments": constraint_arguments,
            }
        )

    def add_driver(self, name: str, target: dict, signal: dict) -> None:
        """
        target:
        - {"type":"coord","body":"B","coord":"x|y|phi"}
        - {"type":"angle","body_i":"...|world","body_j":"...|world"}
        signal:
        - {"kind":"sin"|"cos", "A":float, "omega":float, "phi":0.0, "bias":0.0}
        - {"kind":"linear", "rate":float, "bias":0.0}
            -> sin/cos: f(t) = bias + A * sin(omega * t + phi)   (or cos)
            -> linear:  f(t) = bias + rate * t
        """
        if target["type"] not in ("coord", "angle"):
            raise ValueError("driver target must be 'coord' or 'angle'")
        if signal["kind"] not in ("sin", "cos", "linear"):
            raise ValueError("signal kind must be 'sin', 'cos', or 'linear'")
        self.drivers.append({"name": name, "target": target, "signal": signal})

    def set_gravity(self, gravity: tuple[float, float]) -> None:
        x, y = gravity
        self.options["gravity"] = tuple((float(x), float(y)))

    def compile(self) -> Mechanism:
        float_dtype = jnp.float64
        int_dtype = jnp.int32

        dyn_bodies = [body for body in self.bodies if body.get("dynamic", True)]
        body_id_by_name = {body["name"]: index for index, body in enumerate(dyn_bodies)}
        body_by_name = {body["name"]: body for body in self.bodies}
        
        coordinate_list = []
        coordinate_slices = {}

        for index, body in enumerate(dyn_bodies):
            x_init, y_init, phi_init = body["initial_configuration"]
            coordinate_list.extend([x_init, y_init, phi_init])
            coordinate_slices[body["name"]] = 3 * index, 3 * index + 3
        initial_configuration = jnp.array(coordinate_list, dtype=float_dtype)

        mass = jnp.array([float(body["mass"]) for body in dyn_bodies], dtype=float_dtype)
        inertia_zz = jnp.array([float(body["inertia_zz"]) for body in dyn_bodies], dtype=float_dtype)
        gravity_x, gravity_y = self.options.get("gravity")
        gravity = jnp.array((float(gravity_x), float(gravity_y)), dtype=float_dtype)


        number_of_joints = len(self.joints)
        number_of_bodies = len(dyn_bodies)

        joint_map = {
            "revolute": 0,
            "distance": 1,
            "translation": 2,
            #Below not implemented
            "prismatic": 3,     
            "angle": 4,
            "gear": 5,
            "composite": 6,
            "weld": 7,
        }

        type_to_rows = {
            joint_map["revolute"]: 2,
            joint_map["distance"]: 1,
            joint_map["translation"]: 2, 
            #Below not implemented
            joint_map["prismatic"]: 2,    
            joint_map["angle"]: 1,
            joint_map["weld"]: 3,
            joint_map["composite"]: 0,   
        }

        joint_type_id_list = []
        joint_i_list = []
        joint_j_list = []
        si_list = []
        sj_list = []

        rev_c_list = []               
        dist_value_list = []          
        trans_ui_local_list = []      
        trans_uj_local_list = []

        def _resolve_endpoint(endpoint: dict) -> tuple[int, tuple[float, float]]:
            body_name = endpoint["body"]
            point_name = endpoint["point"]

            if endpoint["body"] != "world" and endpoint["body"] not in body_id_by_name:
                raise ValueError(f"Joint references non-dynamic or unknown body: {endpoint['body']}")
            
            if body_name == "world":
                if point_name not in self.world["points"]:
                    raise ValueError(f"Unknown world point: {point_name}")
                endpoint_x, endpoint_y = self.world["points"][point_name]
                return -1, (float(endpoint_x), float(endpoint_y))

            if body_name not in body_id_by_name:
                raise ValueError(f"Joint references non-dynamic or unknown body: {body_name}")

            body_id = body_id_by_name[body_name]
            points_dict = body_by_name[body_name]["points"]
            if point_name not in points_dict:
                raise ValueError(f"Unknown point '{point_name}' on body '{body_name}'")
            endpoint_x, endpoint_y = points_dict[point_name]

            return body_id, (float(endpoint_x), float(endpoint_y))

        
        for joint in self.joints:
            joint_string = joint["joint_type"].lower().strip()
            type_id = joint_map[joint_string]

            joint_i, si = _resolve_endpoint(joint["endpoint_1"])
            joint_j, sj = _resolve_endpoint(joint["endpoint_2"])

            joint_type_id_list.append(type_id)
            joint_i_list.append(joint_i)
            joint_j_list.append(joint_j)
            si_list.append(si)
            sj_list.append(sj)

            args = joint.get("constraint_arguments", {}) or {}

            if type_id == joint_map["revolute"]:
                # optional world-space offset vector c (defaults to zeros)
                c = args.get("c", (0.0, 0.0))
                rev_c_list.append((float(c[0]), float(c[1])))
                
                #Unused values
                dist_value_list.append(0.0)
                trans_ui_local_list.append((0.0, 0.0))
                trans_uj_local_list.append((0.0, 0.0))

            elif type_id == joint_map["distance"]:
                d = float(args.get("distance", 0.0))
                dist_value_list.append(d)
                rev_c_list.append((0.0, 0.0))
                trans_ui_local_list.append((0.0, 0.0))
                trans_uj_local_list.append((0.0, 0.0))

            elif type_id in (joint_map["translation"], joint_map["prismatic"]):
                ui = args.get("ui_local", (1.0, 0.0))
                uj = args.get("uj_local", (1.0, 0.0))
                trans_ui_local_list.append((float(ui[0]), float(ui[1])))
                trans_uj_local_list.append((float(uj[0]), float(uj[1])))
                rev_c_list.append((0.0, 0.0))
                dist_value_list.append(0.0)

            else:
                #placeholder for unsupported joints
                rev_c_list.append((0.0, 0.0))
                dist_value_list.append(0.0)
                trans_ui_local_list.append((0.0, 0.0))
                trans_uj_local_list.append((0.0, 0.0))
        
        if number_of_joints == 0:
            joint_type_id = jnp.zeros((0,), dtype=int_dtype)
            joint_i = jnp.zeros((0,), dtype=int_dtype)
            joint_j = jnp.zeros((0,), dtype=int_dtype)
            si_xy = jnp.zeros((0, 2), dtype=float_dtype)
            sj_xy = jnp.zeros((0, 2), dtype=float_dtype)
            rev_c_xy = jnp.zeros((0, 2), dtype=float_dtype)                
            dist_value = jnp.zeros((0,), dtype=float_dtype)                
            trans_ui_loc = jnp.zeros((0, 2), dtype=float_dtype)             
            trans_uj_loc = jnp.zeros((0, 2), dtype=float_dtype)  
        else:

            joint_type_id = jnp.array(joint_type_id_list, dtype=int_dtype)
            joint_i = jnp.array(joint_i_list, dtype=int_dtype)
            joint_j = jnp.array(joint_j_list, dtype=int_dtype)
            si_xy = jnp.array(si_list, dtype=float_dtype).reshape(number_of_joints, 2)
            sj_xy = jnp.array(sj_list, dtype=float_dtype).reshape(number_of_joints, 2)
            rev_c_xy = jnp.array(rev_c_list, dtype=float_dtype).reshape(number_of_joints, 2)
            dist_value = jnp.array(dist_value_list, dtype=float_dtype).reshape(number_of_joints,)
            trans_ui_loc = jnp.array(trans_ui_local_list, dtype=float_dtype).reshape(number_of_joints, 2)
            trans_uj_loc = jnp.array(trans_uj_local_list, dtype=float_dtype).reshape(number_of_joints, 2)
            # Normalize with threshold
            trans_ui_loc = trans_ui_loc / jnp.maximum(jnp.linalg.norm(trans_ui_loc, axis=1, keepdims=True), 1e-12)
            trans_uj_loc = trans_uj_loc / jnp.maximum(jnp.linalg.norm(trans_uj_loc, axis=1, keepdims=True), 1e-12)
            


        number_of_coordinates = 3 * number_of_bodies

        if number_of_joints == 0:
            row_counts = jnp.zeros((0,), dtype=int_dtype)
            row_offset = jnp.zeros((0,), dtype=int_dtype)
            number_of_rows = 0

        else:
            row_counts_list = []
            for type_id in joint_type_id_list:
                row_counts_list.append(type_to_rows.get(type_id, 0))
            
            row_counts = jnp.array(row_counts_list, dtype=int_dtype)
            row_offset = jnp.cumsum(row_counts) - row_counts
            number_of_rows = int(row_counts.sum())

        # Driver section
        kind_map = {"sin": 0, "cos": 1, "linear": 2}
        tgt_map  = {"coord": 0, "angle": 1}

        D = len(self.drivers)
        if D == 0:
            drv_target_id = jnp.zeros((0,), dtype=int_dtype)     # 0:coord, 1:angle
            drv_i         = jnp.zeros((0,), dtype=int_dtype)
            drv_j         = jnp.zeros((0,), dtype=int_dtype)
            drv_coord_idx = jnp.zeros((0,), dtype=int_dtype)     # 0:x, 1:y, 2:phi (only for coord)
            sig_kind_id   = jnp.zeros((0,), dtype=int_dtype)     # 0:sin, 1:cos
            sig_params    = jnp.zeros((0, 4), dtype=float_dtype) # (A, omega, phi, bias)
        else:
            _tgt_id, _i, _j, _coord_idx = [], [], [], []
            _kind_id, _params = [], []
            for d in self.drivers:
                tgt = d["target"]; sig = d["signal"]

                # target packing
                _tgt_id.append(tgt_map[tgt["type"]])
                if tgt["type"] == "coord":
                    _i.append(body_id_by_name[tgt["body"]])
                    _j.append(-1)  # unused
                    _coord_idx.append({"x": 0, "y": 1, "phi": 2}[tgt["coord"].lower()])
                else:  # "angle"
                    bi = tgt.get("body_i", "world")
                    bj = tgt.get("body_j", "world")
                    _i.append(-1 if bi == "world" else body_id_by_name[bi])
                    _j.append(-1 if bj == "world" else body_id_by_name[bj])
                    _coord_idx.append(0)  # unused

                # signal packing
                _kind_id.append(kind_map[sig["kind"]])
                if sig["kind"] in ("sin", "cos"):
                    A = float(sig.get("A", 1.0)); w = float(sig.get("omega", 1.0))
                    ph = float(sig.get("phi", 0.0)); b = float(sig.get("bias", 0.0))
                    _params.append((A, w, ph, b))
                else:  # "linear"
                    rate = float(sig.get("rate", 0.0)); b = float(sig.get("bias", 0.0))
                    _params.append((rate, 0.0, 0.0, b))   # (A=rate, omega=0, phi=0, bias=b)

            drv_target_id = jnp.array(_tgt_id, dtype=int_dtype)
            drv_i         = jnp.array(_i, dtype=int_dtype)
            drv_j         = jnp.array(_j, dtype=int_dtype)
            drv_coord_idx = jnp.array(_coord_idx, dtype=int_dtype)
            sig_kind_id   = jnp.array(_kind_id, dtype=int_dtype)
            sig_params    = jnp.array(_params, dtype=float_dtype).reshape(D, 4)

        # rows: each driver contributes one scalar equation
        drv_row_counts     = jnp.ones((D,), dtype=int_dtype) if D > 0 else jnp.zeros((0,), dtype=int_dtype)
        row_counts_all     = jnp.concatenate([row_counts, drv_row_counts], axis=0)
        row_offset_all     = jnp.cumsum(row_counts_all) - row_counts_all
        number_of_rows_all = int(row_counts_all.sum())


        parameters = {
            "mass": mass,
            "inertia_zz": inertia_zz,
            "gravity": gravity,
            "joint_type_id": joint_type_id,
            "joint_i": joint_i,
            "joint_j": joint_j,
            "si_xy": si_xy,
            "sj_xy": sj_xy,
            "types": {
                "revolute": {"c_xy": rev_c_xy},
                "distance": {"d": dist_value},
                "translation": {"ui_local": trans_ui_loc, "uj_local": trans_uj_loc},
            },
            "drivers": {
                "target_id":   drv_target_id,   
                "i":           drv_i,
                "j":           drv_j,
                "coord_index": drv_coord_idx,   
                "signal": {
                    "kind_id": sig_kind_id,     
                    "params":  sig_params,      
                },
            },
        }


        layout = {
            "num_bodies": number_of_bodies,
            "num_joints": number_of_joints,
            "num_coord": number_of_coordinates,
            "body_id_by_name": body_id_by_name,
            "coord_slices": coordinate_slices,
            "row_counts": row_counts_all,
            "row_offset": row_offset_all,
            "num_rows": number_of_rows_all,
            "float_dtype": float_dtype,
            "int_dtype": int_dtype,
            "joint_ids": {
                "revolute": joint_map["revolute"],
                "distance": joint_map["distance"],
                "translation": joint_map["translation"],
                "prismatic": joint_map["prismatic"],
                "angle": joint_map["angle"],
                "weld": joint_map["weld"],
            },
            "driver_ids": {"coord": 0, "angle": 1},
            "signal_ids": {"sin": 0, "cos": 1, "linear": 2},
        }

        report = {
            "num_bodies": number_of_bodies,
            "num_joints": number_of_joints,
            "num_drivers": int(D),
            "num_coord": number_of_coordinates,
            "DoFs": 3 * number_of_bodies - number_of_rows_all,
        }

        return Mechanism(
            parameters=parameters,
            layout=layout,
            q0=initial_configuration,
            report=report,
        )


    def __call__(self, *args, **kwds):
        pass

    def validate_mechanism_setup(self, type:str = "all"):
        pass

    def expose_dict(self, type="world"):
        if type == "world":
            return self.world
        elif type == "bodies":
            return self.bodies
        elif type == "joints":
            return self.joints
        elif type == "drivers":
            return self.drivers
        elif type == "options":
            return self.options
        else:
            print(f"The supplied name was not correct, choose between: bodies, world, joints")





if __name__ == "__main__":
    # -------- Case 1: single body pinned to ground (revolute) --------
    mb = MechanismBuilder("rev_test")
    mb.add_world_point("O", (0.0, 0.0))

    # one dynamic body with two labeled points
    body_pts = {"A": (0.0, 0.0), "B": (1.0, 0.0)}
    mb.add_body("link1", body_pts, initial_configuration=(0.0, 0.0, 0.0), mass=1.0, inertia_zz=0.1)

    # revolute joint at A to world:O, with zero offset c
    mb.add_joint(
        name="J_rev",
        joint_type="revolute",
        endpoint_1={"body": "link1", "point": "A"},
        endpoint_2={"body": "world", "point": "O"},
        constraint_arguments={"c": (0.0, 0.0)},
    )

    params, layout, q0, report = mb.compile()

    print("\n=== Case 1: Revolute to ground ===")
    print("num_bodies:", layout["num_bodies"], " num_joints:", layout["num_joints"])
    print("num_coord:", layout["num_coord"], " num_rows:", layout["num_rows"], " DoFs:", report["DoFs"])
    print("joint_type_id:", params["joint_type_id"])
    print("joint_i:", params["joint_i"], " joint_j:", params["joint_j"])
    print("si_xy[0]:", params["si_xy"][0], " sj_xy[0]:", params["sj_xy"][0])
    print("revolute c_xy[0]:", params["types"]["revolute"]["c_xy"][0])
    print("row_counts:", layout["row_counts"], " row_offset:", layout["row_offset"])
    print("q0:", q0)

    # -------- Case 2: translational joint with non-unit direction (tests normalization) --------
    mb2 = MechanismBuilder("trans_test")
    # two bodies: base and slider
    base_pts = {"P": (0.0, 0.0)}
    slider_pts = {"S": (0.0, 0.0)}
    mb2.add_body("base", base_pts, initial_configuration=(0.0, 0.0, 0.0), mass=1.0, inertia_zz=0.1)
    mb2.add_body("slider", slider_pts, initial_configuration=(0.5, 0.1, 0.0), mass=0.5, inertia_zz=0.05)

    # define a translational joint:
    # - si on base at P (defines the line anchor)
    # - sj on slider at S (constrained point)
    # - ui_local on base is unit x
    # - uj_local on slider is intentionally NON-unit (2,0) to exercise normalization
    mb2.add_joint(
        name="J_trans",
        joint_type="translation",
        endpoint_1={"body": "base", "point": "P"},
        endpoint_2={"body": "slider", "point": "S"},
        constraint_arguments={"ui_local": (1.0, 0.0), "uj_local": (2.0, 0.0)},  # uj_local non-unit on purpose
    )

    params2, layout2, q02, report2 = mb2.compile()

    print("\n=== Case 2: Translational (direction normalization check) ===")
    print("num_bodies:", layout2["num_bodies"], " num_joints:", layout2["num_joints"])
    print("num_coord:", layout2["num_coord"], " num_rows:", layout2["num_rows"], " DoFs:", report2["DoFs"])
    print("joint_type_id:", params2["joint_type_id"])
    print("joint_i:", params2["joint_i"], " joint_j:", params2["joint_j"])
    print("si_xy[0]:", params2["si_xy"][0], " sj_xy[0]:", params2["sj_xy"][0])
    # normalized directions:
    ui = params2["types"]["translation"]["ui_local"][0]
    uj = params2["types"]["translation"]["uj_local"][0]
    print("ui_local (should be unit):", ui)
    print("uj_local (was (2,0), should be unit):", uj)
    # quick norms (visual check)
    import numpy as _np
    print("||ui||:", float(_np.linalg.norm(_np.asarray(ui))))
    print("||uj||:", float(_np.linalg.norm(_np.asarray(uj))))
    print("row_counts:", layout2["row_counts"], " row_offset:", layout2["row_offset"])
    print("q0:", q02)
